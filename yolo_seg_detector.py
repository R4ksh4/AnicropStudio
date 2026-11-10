# -*- coding: utf-8 -*-
"""
yolo_seg_detector.py - YOLO 实例分割与物理暗影比提取引擎 (高灵敏度防漏检 + 模特边缘阻断)
"""

import os
import urllib.request
from pathlib import Path
import cv2
import numpy as np


class YOLOSegDetector:
    _session = None

    @classmethod
    def _download_model_file(cls, target_path: Path) -> bool:
        """多节点免登录可靠下载 YOLOv8n-seg 实例分割权重 (约 13.9MB)"""
        urls = [
            "https://hf-mirror.com/unileon-robotics/YOLO8-ONNX/resolve/main/yolov8n-seg.onnx",
            "https://hf-mirror.com/EdgeFirst/yolov8-seg/resolve/main/onnx/yolov8n-seg-coco.onnx",
            "https://huggingface.co/unileon-robotics/YOLO8-ONNX/resolve/main/yolov8n-seg.onnx"
        ]
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
        }

        tmp_path = target_path.with_suffix(".onnx.tmp")
        for u in urls:
            try:
                print(f"⚡ [YOLO-seg] 正在从公开节点拉取实例分割模型 (约 13.9MB):\n   -> {u}")
                req = urllib.request.Request(u, headers=headers)
                with urllib.request.urlopen(req, timeout=30) as resp, open(tmp_path, "wb") as f:
                    while True:
                        buf = resp.read(64 * 1024)
                        if not buf:
                            break
                        f.write(buf)

                if tmp_path.exists() and tmp_path.stat().st_size > 10 * 1024 * 1024:
                    os.replace(tmp_path, target_path)
                    print("✅ [YOLO-seg] 实例分割权重下载完毕！")
                    return True
            except Exception as e:
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except Exception:
                        pass
        return False

    @classmethod
    def _get_session(cls):
        if cls._session is None:
            import onnxruntime as ort

            base_dir = Path(__file__).resolve().parent
            assets_dir = base_dir / "assets"
            assets_dir.mkdir(parents=True, exist_ok=True)

            model_path = assets_dir / "yolov8n-seg.onnx"
            if not model_path.exists() or model_path.stat().st_size < 10 * 1024 * 1024:
                ok = cls._download_model_file(model_path)
                if not ok or not model_path.exists():
                    raise RuntimeError(f"无法下载 YOLO-seg 权重，请手动下载放入: {model_path}")

            avail = ort.get_available_providers()
            providers = []
            if 'CUDAExecutionProvider' in avail:
                providers.append('CUDAExecutionProvider')
            providers.append('CPUExecutionProvider')

            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = min(6, os.cpu_count() or 4)

            cls._session = ort.InferenceSession(str(model_path), sess_options=opts, providers=providers)

        return cls._session

    @classmethod
    def segment_people(
        cls,
        orig_bgr: np.ndarray,
        conf_thresh: float = 0.06,  # 灵敏度放宽到 0.06，右侧被严重遮挡的半截小脚一网打尽
        iou_thresh: float = 0.45
    ) -> tuple[list[dict], dict | None]:
        orig_h, orig_w = orig_bgr.shape[:2]
        session = cls._get_session()

        in_tensor = session.get_inputs()[0]
        in_shape = in_tensor.shape
        input_h = in_shape[2] if (len(in_shape) > 2 and isinstance(in_shape[2], int) and in_shape[2] > 0) else 480
        input_w = in_shape[3] if (len(in_shape) > 3 and isinstance(in_shape[3], int) and in_shape[3] > 0) else 640

        scale = min(input_w / float(orig_w), input_h / float(orig_h))
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        resized = cv2.resize(orig_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
        pad_x = (input_w - new_w) // 2
        pad_y = (input_h - new_h) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 255.0, (input_w, input_h), swapRB=True)
        net_outs = session.run([o.name for o in session.get_outputs()], {in_tensor.name: blob})

        preds = np.squeeze(net_outs[0])
        if preds.shape[0] == 116:
            preds = preds.T

        proto_masks = np.squeeze(net_outs[1])
        num_proto, proto_h, proto_w = proto_masks.shape

        boxes = []
        confidences = []
        mask_coeffs = []

        for row in preds:
            scores = row[4:84]
            person_score = scores[0]
            if person_score >= conf_thresh:
                cx, cy, bw, bh = row[0], row[1], row[2], row[3]
                x1 = (cx - bw / 2.0 - pad_x) / scale
                y1 = (cy - bh / 2.0 - pad_y) / scale
                w_orig = bw / scale
                h_orig = bh / scale

                boxes.append([int(x1), int(y1), int(w_orig), int(h_orig)])
                confidences.append(float(person_score))
                mask_coeffs.append(row[84:116])

        if not boxes:
            return [], None

        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=conf_thresh, nms_threshold=iou_thresh)
        if len(indices) == 0:
            return [], None

        detected_people = []
        proto_flat = proto_masks.reshape(num_proto, -1)

        for i in np.array(indices).flatten():
            bx, by, bw, bh = boxes[i]
            bx = max(0, min(orig_w - 1, bx))
            by = max(0, min(orig_h - 1, by))
            bw = max(1, min(orig_w - bx, bw))
            bh = max(1, min(orig_h - by, bh))

            coeff = mask_coeffs[i]
            mask_hw = np.matmul(coeff, proto_flat).reshape(proto_h, proto_w)
            mask_hw = 1.0 / (1.0 + np.exp(-mask_hw))

            crop_x1 = max(0, int(pad_x * proto_w / input_w))
            crop_y1 = max(0, int(pad_y * proto_h / input_h))
            crop_w = min(proto_w - crop_x1, int(new_w * proto_w / input_w))
            crop_h = min(proto_h - crop_y1, int(new_h * proto_h / input_h))
            mask_cropped = mask_hw[crop_y1:crop_y1 + crop_h, crop_x1:crop_x1 + crop_w]

            mask_full = cv2.resize(mask_cropped, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            bin_mask = (mask_full > 0.45).astype(np.uint8) * 255

            bbox_gate = np.zeros((orig_h, orig_w), dtype=np.uint8)
            bbox_gate[by:by + bh, bx:bx + bw] = 255
            bin_mask = cv2.bitwise_and(bin_mask, bbox_gate)

            area = cv2.countNonZero(bin_mask)
            if area > 200:
                detected_people.append({
                    "id": len(detected_people),
                    "score": confidences[i],
                    "bbox": (bx, by, bw, bh),
                    "mask": bin_mask,
                    "area": area,
                    "is_subject": False
                })

        if not detected_people:
            return [], None

        # 主角模特智能仲裁
        img_cx = orig_w / 2.0
        def _subject_affinity(p):
            bx, by, bw, bh = p["bbox"]
            obj_cx = bx + bw / 2.0
            dist_norm = 1.0 - (abs(obj_cx - img_cx) / img_cx)
            return p["area"] * (dist_norm ** 1.6)

        detected_people.sort(key=_subject_affinity, reverse=True)
        detected_people[0]["is_subject"] = True
        subject = detected_people[0]
        passersby = [p for p in detected_people if not p["is_subject"]]

        for idx, p in enumerate(passersby):
            p["id"] = idx

        return passersby, subject

    @classmethod
    def absorb_feet_shadow_photometric(
        cls,
        person_mask: np.ndarray,
        orig_bgr: np.ndarray,
        patch_aligned: np.ndarray
    ) -> np.ndarray:
        """
        【物理暗影比光学提取算子】
        原理：地面投影本质是明度线性衰减，色相保持一致。
        以鞋底接触面为种子点，顺着明度差分下潜吞噬影子，绝不误伤金属接缝。
        """
        orig_h, orig_w = orig_bgr.shape[:2]
        merged_mask = person_mask.copy()

        y_indices, x_indices = np.where(person_mask > 0)
        if len(y_indices) == 0:
            return merged_mask

        feet_y = np.max(y_indices)
        person_h = feet_y - np.min(y_indices)
        if person_h < 30:
            return merged_mask

        # 转换到 Lab 空间精准提取亮度明度通道 (Luminance)
        orig_lab = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2Lab)
        patch_lab = cv2.cvtColor(patch_aligned, cv2.COLOR_BGR2Lab)

        orig_l = orig_lab[:, :, 0].astype(np.float32)
        patch_l = patch_lab[:, :, 0].astype(np.float32)

        # 影子区域满足：原图明显比 AI 补丁暗 (patch_l > orig_l + 8) 且 色差保持极小
        lum_diff = np.clip(patch_l - orig_l, 0, 255).astype(np.uint8)

        shadow_box_h = int(person_h * 0.40)
        min_x = max(0, np.min(x_indices) - 40)
        max_x = min(orig_w, np.max(x_indices) + 40)
        start_y = max(0, feet_y - 12)
        end_y = min(orig_h, feet_y + shadow_box_h)

        floor_lum = np.zeros((orig_h, orig_w), dtype=np.uint8)
        floor_lum[start_y:end_y, min_x:max_x] = lum_diff[start_y:end_y, min_x:max_x]

        # 结合原图高频 Sobel 边缘阻断，严禁影子算子跨越金属接缝和踢脚线
        gray = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
        abs_x = cv2.convertScaleAbs(sobel_x)
        abs_y = cv2.convertScaleAbs(sobel_y)
        edge_mag = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
        # 强边缘处作为物理阻断墙 (Barrier)
        floor_lum[edge_mag > 45] = 0

        _, shadow_bin = cv2.threshold(floor_lum, 10, 255, cv2.THRESH_BINARY)
        kernel_connect = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 23))
        shadow_fused = cv2.morphologyEx(shadow_bin, cv2.MORPH_CLOSE, kernel_connect)

        contours, _ = cv2.findContours(shadow_fused, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cy = cnt[:, 0, 1]
            bx, by, bw, bh = cv2.boundingRect(cnt)
            aspect = max(bw, bh) / float(max(1, min(bw, bh)))
            # 避让狭长金属线
            if aspect > 5.0 and min(bw, bh) < 20:
                continue

            if np.min(cy) <= feet_y + 12 and cv2.contourArea(cnt) > 120:
                cv2.drawContours(merged_mask, [cnt], -1, 255, thickness=-1)

        return merged_mask