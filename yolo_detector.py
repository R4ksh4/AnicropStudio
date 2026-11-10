# -*- coding: utf-8 -*-
"""
yolo_detector.py - Anicrop Studio YOLO 语义目标识别引擎 (纯 ONNX Runtime，无 PyTorch 依赖)
专用于漫展照片路人 (Person) 与随身杂物 (Bags/Bottles) 语义级定位
"""

import os
import shutil
import urllib.request
from pathlib import Path
import cv2
import numpy as np

# COCO 80 类别中需要关注的漫展消除目标 (重点抓人和常见杂物)
TARGET_CLASSES = {
    0: "路人",        # person
    24: "背包",       # backpack
    26: "手提包",     # handbag
    28: "旅行箱",     # suitcase
    39: "瓶子/水杯",  # bottle
    56: "椅子/杂物",  # chair
}


class YOLODetector:
    _session = None

    @classmethod
    def _download_model_file(cls, target_path: Path) -> bool:
        """带浏览器请求头的多节点可靠下载器"""
        urls = [
            # 节点 1: 国内极速镜像 (无需梯子，免配置秒拉)
            "https://hf-mirror.com/SpotLab/YOLOv8Detection/resolve/main/yolov8n.onnx",
            # 节点 2: Hugging Face 官方直链
            "https://huggingface.co/SpotLab/YOLOv8Detection/resolve/main/yolov8n.onnx",
            # 节点 3: GitHub 极速备用
            "https://github.com/yoobright/yolo-onnx/raw/master/yolov8n.onnx"
        ]
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        tmp_path = target_path.with_suffix(".onnx.tmp")
        for u in urls:
            try:
                print(f"⚡ [YOLODetector] 正在从高速节点拉取 YOLOv8n (约 12MB):\n   -> {u}")
                req = urllib.request.Request(u, headers=headers)
                with urllib.request.urlopen(req, timeout=30) as resp, open(tmp_path, "wb") as f:
                    while True:
                        buf = resp.read(64 * 1024)
                        if not buf:
                            break
                        f.write(buf)

                if tmp_path.exists() and tmp_path.stat().st_size > 5 * 1024 * 1024:
                    os.replace(tmp_path, target_path)
                    print("✅ [YOLODetector] 权重下载完毕并校验通过！")
                    return True
            except Exception as e:
                print(f"⚠️ [YOLODetector] 节点响应异常 ({e})，自动切换备用镜像...")
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

            model_path = assets_dir / "yolov8n.onnx"
            # 检查文件完整性 (小于 5MB 视为损坏或残废)
            if not model_path.exists() or model_path.stat().st_size < 5 * 1024 * 1024:
                ok = cls._download_model_file(model_path)
                if not ok or not model_path.exists():
                    raise RuntimeError(
                        f"无法自动下载 YOLOv8n 模型，请手动下载并放入:\n{model_path}\n"
                        f"推荐直链: https://hf-mirror.com/SpotLab/YOLOv8Detection/resolve/main/yolov8n.onnx"
                    )

            avail = ort.get_available_providers()
            providers = []
            if 'CUDAExecutionProvider' in avail:
                providers.append('CUDAExecutionProvider')
            providers.append('CPUExecutionProvider')

            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = min(6, os.cpu_count() or 4)

            cls._session = ort.InferenceSession(str(model_path), sess_options=opts, providers=providers)
            print(f"🚀 [YOLODetector] YOLO 语义识别引擎启动！执行端: 【{cls._session.get_providers()[0]}】")

        return cls._session

    @classmethod
    def detect_objects(cls, bgr_img: np.ndarray, conf_thresh: float = 0.20, iou_thresh: float = 0.45) -> list[dict]:
        """
        输入任意分辨率的原图 BGR，输出结构化实体列表:
        [{"id": 0, "class_id": 0, "label": "路人", "score": float, "bbox": (x, y, w, h), "is_subject": bool}, ...]
        """
        orig_h, orig_w = bgr_img.shape[:2]
        session = cls._get_session()

        # 1. 经典 640 Letterbox 等比缩放
        input_size = 640
        scale = min(input_size / float(orig_w), input_size / float(orig_h))
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        resized = cv2.resize(bgr_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
        pad_x = (input_size - new_w) // 2
        pad_y = (input_size - new_h) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        # 2. 前向推理 (1x3x640x640 float32)
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 255.0, (input_size, input_size), swapRB=True)
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: blob})[0]  # shape: (1, 84, 8400)

        # 3. 维度转置为 (8400, 84)
        preds = np.squeeze(outputs).T
        boxes = []
        confidences = []
        class_ids = []

        for row in preds:
            scores = row[4:]
            class_id = np.argmax(scores)
            score = scores[class_id]

            if score >= conf_thresh and class_id in TARGET_CLASSES:
                cx, cy, bw, bh = row[0], row[1], row[2], row[3]
                x1 = (cx - bw / 2.0 - pad_x) / scale
                y1 = (cy - bh / 2.0 - pad_y) / scale
                w_orig = bw / scale
                h_orig = bh / scale

                boxes.append([int(x1), int(y1), int(w_orig), int(h_orig)])
                confidences.append(float(score))
                class_ids.append(int(class_id))

        if not boxes:
            return []

        # 4. NMS 非极大值抑制
        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=conf_thresh, nms_threshold=iou_thresh)
        if len(indices) == 0:
            return []

        detected = []
        for i in np.array(indices).flatten():
            bx, by, bw, bh = boxes[i]
            bx = max(0, min(orig_w - 1, bx))
            by = max(0, min(orig_h - 1, by))
            bw = max(1, min(orig_w - bx, bw))
            bh = max(1, min(orig_h - by, bh))

            cid = class_ids[i]
            detected.append({
                "id": len(detected),
                "class_id": cid,
                "label": TARGET_CLASSES.get(cid, "路人"),
                "score": confidences[i],
                "bbox": (bx, by, bw, bh),
                "area": bw * bh,
                "is_subject": False
            })

        # 5. 主角模特智能锁定：面积最大且靠近画面中轴的 Person 标为主体
        people = [d for d in detected if d["class_id"] == 0]
        if people:
            img_cx = orig_w / 2.0
            def _subject_affinity(d):
                bx, by, bw, bh = d["bbox"]
                obj_cx = bx + bw / 2.0
                dist_norm = 1.0 - (abs(obj_cx - img_cx) / img_cx)
                return d["area"] * (dist_norm ** 1.5)

            people.sort(key=_subject_affinity, reverse=True)
            people[0]["is_subject"] = True
            people[0]["label"] = "主体模特 (已锁定保护)"

        return detected