import os
import sys

import cv2
import numpy as np
import onnxruntime as ort
import PySide6
import scipy


def main():
    print(f"Python: {sys.version.split()[0]}")
    print(f"OpenCV: {cv2.__version__} | NumPy: {np.__version__} | SciPy: {scipy.__version__}")
    print(f"PySide6: {PySide6.__version__} | ORT: {ort.__version__}")

    # 预加载 pip 安装的 CUDA/cuDNN DLL，防止 Windows 找不到路径
    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls(cuda=True, cudnn=True, msvc=True)
        except Exception as e:
            print(f"[提示] preload_dlls 跳过: {e}")

    print("系统可用推理后端:", ort.get_available_providers())

    model_path = os.path.join("models", "depth_anything_v2_vits.onnx")
    if not os.path.exists(model_path):
        print(f"[警告] 未找到模型文件: {model_path}，请先下载到 models/ 目录")
        return

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(model_path, providers=providers)
    print(f"模型加载成功！当前实际启用后端: {sess.get_providers()}")

    for inp in sess.get_inputs():
        print(f"  输入节点 -> {inp.name}: shape={inp.shape}, type={inp.type}")
    for out in sess.get_outputs():
        print(f"  输出节点 -> {out.name}: shape={out.shape}, type={out.type}")


if __name__ == "__main__":
    main()