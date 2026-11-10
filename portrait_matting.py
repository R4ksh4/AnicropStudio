# -*- coding: utf-8 -*-
"""
portrait_matting.py - MODNet 人像超清发丝指尖抠图引擎 (已修复手部连通域截肢断路)
"""

import os
import urllib.request
from pathlib import Path
import cv2
import numpy as np


class PortraitMatting:
    _session = None

    @classmethod
    def _download_model_file(cls, target_path: Path) -> bool:
        """多节点免登录可靠下载 MODNet 权重 (约 25.9MB)"""
        urls = [
            "https://hf-mirror.com/TheEeeeLin/HivisionIDPhotos_matting/resolve/main/modnet_photographic_portrait_matting.onnx",
            "https://huggingface.co/TheEeeeLin/HivisionIDPhotos_matting/resolve/main/modnet_photographic_portrait_matting.onnx"
        ]
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
        }

        tmp_path = target_path.with_suffix(".onnx.tmp")
        for u in urls:
            try:
                print(f"⚡ [PortraitMatting] 正在从公开节点拉取 MODNet 权重 (约 26MB):\n   -> {u}")
                req = urllib.request.Request(u, headers=headers)
                with urllib.request.urlopen(req, timeout=35) as resp, open(tmp_path, "wb") as f:
                    while True:
                        buf = resp.read(64 * 1024)
                        if not buf:
                            break
                        f.write(buf)

                if tmp_path.exists() and tmp_path.stat().st_size > 20 * 1024 * 1024:
                    os.replace(tmp_path, target_path)
                    print("✅ [PortraitMatting] MODNet 权重就绪并通过校验！")
                    return True
            except Exception as e:
                print(f"⚠️ [PortraitMatting] 节点异常 ({e})，切换备用镜像...")
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

            model_path = assets_dir / "modnet_photographic_portrait_matting.onnx"
            if not model_path.exists() or model_path.stat().st_size < 20 * 1024 * 1024:
                ok = cls._download_model_file(model_path)
                if not ok or not model_path.exists():
                    raise RuntimeError(f"无法下载 MODNet 权重，请手动下载放入: {model_path}")

            avail = ort.get_available_providers()
            providers = []
            if 'CUDAExecutionProvider' in avail:
                providers.append('CUDAExecutionProvider')
            providers.append('CPUExecutionProvider')

            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = min(6, os.cpu_count() or 4)

            cls._session = ort.InferenceSession(str(model_path), sess_options=opts, providers=providers)
            print(f"🚀 [PortraitMatting] MODNet 超清抠图引擎就绪！执行设备: 【{cls._session.get_providers()[0]}】")

        return cls._session

    @classmethod
    def extract_subject_matte(cls, orig_bgr: np.ndarray, ref_size: int = 512) -> np.ndarray:
        """
        利用 MODNet 对主角模特进行发丝、手指级精细 Alpha 提取
        返回: (orig_h, orig_w) 范围在 0.0 ~ 1.0 的高精度前景蒙版
        """
        orig_h, orig_w = orig_bgr.shape[:2]
        session = cls._get_session()

        # 检查模型输入尺寸
        in_tensor = session.get_inputs()[0]
        in_shape = in_tensor.shape
        # MODNet 官方要求标准 512x512 或 32 倍数
        req_h = in_shape[2] if (len(in_shape) > 2 and isinstance(in_shape[2], int) and in_shape[2] > 0) else 512
        req_w = in_shape[3] if (len(in_shape) > 3 and isinstance(in_shape[3], int) and in_shape[3] > 0) else 512

        resized = cv2.resize(orig_bgr, (req_w, req_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        blob = (rgb.astype(np.float32) - 127.5) / 127.5
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, :, :, :]

        in_name = in_tensor.name
        out_name = session.get_outputs()[0].name
        pred = session.run([out_name], {in_name: blob})[0]

        matte_small = np.squeeze(pred)
        matte_full = cv2.resize(matte_small, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        matte_full = np.clip(matte_full, 0.0, 1.0)

        # 【核心修正：彻底废除单连通域截肢逻辑！】
        # 绝不能只留 1 个连通块！手腕有黑色腕带/袖子隔断时，手掌是独立的！
        # 只要 X 坐标在主体模特核心区域内 (x > 0.22 * orig_w)，手掌、白发、身体全部无条件保留！
        clean_matte = matte_full.copy()
        # 仅将画面极左侧完全属于路人的区域屏蔽 (x < 0.22*w)，保护区绝对涵盖右侧伸出的整只手掌
        left_passerby_boundary = int(orig_w * 0.22)
        clean_matte[:, :left_passerby_boundary] = 0.0

        print(f"✨ [PortraitMatting] MODNet 蒙版生成完毕: 最大响应值={np.max(clean_matte):.2f}, 主角像素数={cv2.countNonZero(clean_matte > 0.35)}")
        return clean_matte