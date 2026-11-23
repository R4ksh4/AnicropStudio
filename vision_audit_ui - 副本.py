# -*- coding: utf-8 -*-
"""
vision_audit_ui.py - Anicrop Studio 多模态审美审计实验台 (独立可运行 / 可无缝内嵌 ui.py)
特性:
  1. 支持拖拽/加载照片 (JPG/PNG/CR3/ARW/NEF) + 官方参考立绘；
  2. 自动复用 album.py & FocusInspector 生成 EXIF 遥测 + 多尺度 1:1 ROI 切片并实时预览；
  3. 支持加载/执行自定义 Python 前置分析脚本 (run_hook(context) -> dict)；
  4. 支持可视化编辑 System / Step 1~9 / Pass 2 / Pass 3 提示词并导入导出 JSON；
  5. 并发 9 步状态机 (0 -> 1 -> 2)，每步均强制带图看图，支持对存疑点向摄影师反问并二次校准。
"""

import os
import sys
import json
import time
import base64
import importlib.util
import urllib.request
import urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QTextEdit, QPlainTextEdit, QFrame, QGroupBox, QSplitter, QTabWidget,
    QFileDialog, QMessageBox, QComboBox, QSpinBox, QScrollArea, QApplication, QCheckBox
)
from PySide6.QtCore import Qt, Signal, QThread, QSettings
from PySide6.QtGui import QPixmap, QImage, QDragEnterEvent, QDropEvent, QFont, QTextCursor

# 尝试复用 AnicropStudio 现有底层轮子（独立运行或目录缺失时优雅降级）
try:
    from album import load_photo_qimage, parse_photo_exif, extract_af_points, get_app_cache_root, is_raw_file
    HAS_ALBUM = True
except Exception:
    HAS_ALBUM = False

try:
    from focus_inspector import FocusInspector
    HAS_FOCUS_INSPECTOR = True
except Exception:
    HAS_FOCUS_INSPECTOR = False


# ==============================================================================
# 默认提示词模板 (v1.0)
# ==============================================================================
DEFAULT_SYSTEM_PROMPT = """你是「Anicrop Vision Auditor」，一名拥有十年漫展实战与商业人像后期经验的资深二次元摄影审片总监。
你的核心职责是：结合物理遥测数据与多尺度图像切片，对漫展 Cosplay 照片进行客观、冷酷、符合人类摄影师直觉的专业审计。

必须严格遵守以下铁律：
1. 【零彩虹屁原则】：严禁使用“唯美、惊艳、不错、很有氛围感”等空洞赞美词。优点必须落到具体光学、色彩、神态或结构依据上；瑕疵必须精确指出部位与成因。
2. 【视觉实证原则】：你看到的每一条结论都必须以当前输入图片（全景图 / 1:1 局部切片 / 官方参考图）中肉眼可见的像素事实为唯一依据，严禁脑补画面中不存在或未入镜的部位。
3. 【遥测融合原则】：如果输入中提供了 EXIF 物理参数（焦段、快门、ISO、人脸像素占比、锐度值），判断“糊片/畸变/过曝”时必须结合物理上限综合判定，禁止脱离像素占比空谈模糊。
4. 【未入镜诚实原则】：若当前步骤检查的部位在画面中完全未入镜（如大头照没有手或脚），将 "applicable" 设为 false，绝不强行编造。
5. 【严格 JSON 输出】：只输出合法的纯 JSON 对象，不要包含任何多余闲聊文字。"""

DEFAULT_STEPS = [
    (1, "构图与场景", ["full"],
     "1. 【拍摄场景定性】：若遥测数据中已由摄影师明确指定了拍摄场景（非“自动识别”），直接采信该场景定性并结合画面特征做细化补充；若为“自动识别”，则根据背景、地面、设备与光影特征判定属于【官方展台/官方集邮点/漫展摄影区/漫展公共通道/专业棚拍/室内实景/户外外景】并写明依据。\n"
     "2. 【景别与占比】：判定景别（特写/半身/七分身/全身）与横竖画幅，评估人物面积占比与位置。\n"
     "3. 【三层倾斜与画面失衡必检项（禁止跳过）】：以照片上下左右四条边框为绝对水平/垂直基准（哪怕背景高度虚化，也要观察前景台面/灯带斜向、背景色块斜向、以及人物脊柱/双肩连线相对于画框的夹角）。"
     "你必须在 visual_evidence 中专门写出一条以 `【三层倾斜与重心】` 开头的话，明确交代：①前景倾斜度、②背景倾斜度、③人物倾斜度、④画面是否存在明显往一侧滑落的失衡感。"
     "随后判断倾斜是否合理：若倾斜服务于强动态张力且重心稳固则合理；若人物偏静态摆拍、却因机位拧斜导致前景/背景斜切压迫主体、产生明显的失衡滑落感，必须高权重列为 flaws 核心构图瑕疵（勿美化为动感），并评估旋转拉正是否切伤主体。\n"
     "4. 【切线与杂物】：检查脚部、手腕、头顶饰物的死亡截断；结合场景属性（棚拍零容忍，展台/通道按干扰度分级）扫描背景杂物、路人及器材穿帮。"),
    (2, "角色与脸脖衔接", ["full", "face_neck", "ref"],
     "1. 对比角色设定与官方参考图（如有），检查服装版型、假发颜色/样式、发饰/道具佩戴位置是否还原，有无明显 OOC。\n"
     "2. 重点盯着【1:1 脸脖切片】中的下颌线至脖子/锁骨过渡区：判断肤色是否断层？如果脖子比脸暗/黄，先客观记录颜色差异与边界特征（注意区分底粉色号过白还是主光角度导致的阴影，Pass 2 会与布光步骤交叉复核）。"),
    (3, "色彩影调与畸变", ["full"],
     "1. 结合 EXIF 曝光三要素与直方图分布，检查是否存在过曝死白（尤其面部高光）或欠曝死黑。\n"
     "2. 检查饱和度是否过饱和溢色或欠饱和发灰；严格区分“刻意黑白片/风格化调色”与“失败的脏色/白平衡失控”。\n"
     "3. 结合 EXIF 镜头型号与物理焦段，检查广角透视或桶形/枕形畸变：正常范围提一句即可；若面部、四肢或背景明显拉扯变形，指出需进 LR/像素蛋糕套镜头配置文件纠正。"),
    (4, "动作与体态", ["full", "body", "ref"],
     "1. 评估肢体动作与该角色性格设定是否贴合（有无 OOC 或僵硬感）。\n"
     "2. 检查体态力学瑕疵：手臂紧贴躯干压扁显壮（麒麟臂）、耸肩没脖子、低头双下巴或重心怪异。"),
    (5, "人脸与妆造(R)", ["face_neck", "full", "ref"],
     "1. 【清晰度铁律】：结合遥测中的人脸像素宽高与快门速度，判断眼部/面部是否脱焦或手抖糊片（若全身照人脸仅占极少像素且快门安全，不得误判脱焦）。\n"
     "2. 【细节显微镜】：逐项检查卡粉/浮粉、鼻贴/双眼皮贴反光穿帮、卧蚕、唇妆、假睫毛翘边。\n"
     "3. 【状态与修图审计】：客观记录法令纹、木偶纹、苹果肌挤压、痘痘或闭眼/半睁眼；客观识别脸型（瓜子脸/圆脸等）与胖瘦美丑；检查妆造 OOC。"
     "【磨皮与液化低权重铁律】：漫展 Coser 普遍使用高遮瑕厚底妆与定妆粉，叠加柔光灯后脸部无毛孔、无瑕疵属于正常物理妆效，严禁仅凭“皮肤光滑/无毛孔”就判磨皮过度！仅当出现眼睫毛/鼻翼/唇线边缘被抹糊、五官明显失真、或背景直线被液化拉弯时，才作为低~中权重问题提示。"),
    (6, "头发与发际线", ["hairline", "full", "ref"],
     "1. 检查发际线、鬓角与后颈边缘：假发发网是否露馅、真发碎发是否漏出、假发分缝是否炸开。\n"
     "2. 检查头发整体杂乱飞丝程度、假发是否压塌、发色与发型是否符合角色设定（有无 OOC）。"),
    (7, "手部专项", ["body", "full"],
     "1. 若双手均未入镜，直接标 applicable=false。\n"
     "2. 结合快门速度检查手部是否运动模糊。\n"
     "3. 评估手指与手腕动作是否协调自然（有无鸡爪手、手指正对镜头穿刺、手部发暗发紫等问题）。"),
    (8, "布光公式反推", ["full", "face_neck"],
     "1. 结合瞳孔眼神光形状与阴影走向，反推现场用了几盏灯（如单灯直闪、1主+1轮廓等）以及主光方向。\n"
     "2. 评估现场环境光与补光灯的杂乱度（有无漫展顶灯绿/紫毒光污染、色温打架）。\n"
     "3. 检查发丝轮廓光是否出来、光影方向是否合理、画面边缘或反光面有无灯架/灯头穿帮。"),
    (9, "身形与穿帮专项", ["body", "full"],
     "1. 客观评估二次元身形视觉表现（胸腰臀比例、腹部线条、是否显得有钢板感或赘肉挤压）。\n"
     "2. 若拍到腿部/足部/丝袜/赤足，检查露出状态与透视是否自然；未拍到则注明。\n"
     "3. 严查服装穿帮：安全裤露出、内衣肩带/隐形肩带反光、胸垫/垫片（饭盒）显形穿帮、拉链崩开等。"),
]

DEFAULT_PASS2_PROMPT = """【当前任务】：执行 Pass 2 —— 带图交叉复核（Cross-Check），将第 {step_id} 步（{step_name}）从 status=1 推进到 status=2。
【核心铁律】：你必须重新结合当前挂载的原图与 1:1 切片亲自确认像素事实，绝不允许只看文字盲目盖章！

【物理遥测与现场背景】：{EXIF_TELEMETRY}
【Pass 1 全部9步初检结果汇总】：
{ALL_9_STEP_JSONS}

【交叉复核强制动作】：
1. 拿其他步骤的证据挑战或印证 Step {step_id} 的结论：
   - 【场景尺度校准】：结合 Step 1 判定的拍摄场景校准容错度（棚拍出现杂物/乱光/穿帮从严重判；官方展台/漫展通道的环境设备与路人则重点评估对主体的遮挡破坏度与后期可擦除成本）。
   - 【斜构图与失衡必复核（高权重）】：重新盯着全景图四条画框边缘，核查 Step 1 是否漏检了整张图的机位倾斜（尤其注意前景台面、边缘灯带与人物躯干轴线是否明显斜着滑向一侧）。若画面存在明显倾斜且结合 Step 4 并非强动态瞬间、导致视觉重心失衡，哪怕有前景遮挡等其他大问题，也绝不能放过斜构图失衡，必须作为高权重核心构图瑕疵列进 flaws。
   - 【其余因果消歧】：脸脖色差是光影阴影还是粉底？人脸糊不糊是否受像素占比与快门支持？若 Pass 1 指控“磨皮/液化过重”，结合漫展厚底妆常态与人脸像素占比复核——只要睫毛、唇线、发丝边缘依然锐利且背景无拉扯变形，直接撤回或降为极低权重提示；神态+手势+角色性格是否统一？暗调/黑白是风格化还是欠曝？
2. 若发现 Pass 1 某条判断误杀或看漏，必须在 "pass2_cross_check_notes" 写明推翻/修正理由，并更新 strengths/flaws。
3. 若某点按常规像失误但可能是摄影师刻意创作选择，放入 "ambiguous_for_user" 生成尖锐反问。
4. 只输出最终 JSON，且必须将 "status" 设为 2。"""

DEFAULT_PASS3_PROMPT = """【当前任务】：所有 9 步已完成 Pass 2 带图交叉核验（全员 status=2）。请结合挂载的全景图与下面 9 份终审 JSON，输出资深漫展摄影审片总监视角的最终报告。

【物理遥测与现场背景】：{EXIF_TELEMETRY}
【角色信息】：{CHARACTER_INFO}
【全部 9 步终审 JSON】：
{ALL_9_VERIFIED_JSONS}

请严格按以下 Markdown 结构输出（直击要害，禁止空洞客套）：
### 🎯 场景定性与一句话结论
（先标明【拍摄场景：xxx（附依据）】，再给出【定性：直接进精选精修 / 局部抢救可出片 / 建议进废片篓】，并一句话点破核心成败点；若确认存在不合理斜构图导致画面失衡，须在结论和核心瑕疵中重点点名）

### ✅ 客观亮点
（只列真正成立的光学、色彩、神态、构图或还原度亮点）

### ⚠️ 客观瑕疵与后期抢救成本
（按【致命不可逆】、【后期低成本可救（附 LR/PS/AI消除思路）】、【微小瑕不掩瑜】分级列出）

### 🧠 审美与风格主观锐评（结合器材性能与资历动态评判）
（结合本次实际机身、镜头参数、摄影师资历与现场可控条件，先评价情绪张力、角色灵魂还原度与成片潜力。
【动态器材反差反应原则 — 严禁套用固定台词，必须完全针对当前具体器材与画面事实现场发挥】：
- 若用入门机身/半幅/套头/老单反等性价比器材，在极限条件下榨出了极高的锐度、宽容度或抓拍水准，毫不吝啬地给出具体到像素与参数细节的惊叹与赞赏；
- 若手握顶级旗舰机身或大三元/大定焦等昂贵器材，却拍出了低级跑焦、曝光崩坏或灾难性构图，允许先用符合极客摄影师性格的犀利吐槽开喷一句（紧扣具体失误与具体器材），随后立刻注明“开个玩笑，说正经的：”并切回冷酷客观的技术分析；
- 若器材与成片水准正常匹配，则保持冷静专业点评，不强行玩梗。
同时注意：若遥测声明老师或灯光不可控，【存疑点】与【建议】中严禁出现让模特重摆或改灯位的废话。）

### ❓ 需要你亲自确认的存疑点（避免误判）
（将 9 步中的存疑点去重、合并同类项，只挑出最核心的 3~5 个问题，编号 1. 2. 3. 直接问摄影师“这里你为什么这么拍/修？”，严禁啰嗦凑数；若无存疑点则写“无存疑点，事实清晰”。）

【重要附带输出】：在上面整篇 Markdown 报告全部写完后，另起一行严格输出分隔符 `===QUESTIONS_JSON===`，紧接着输出一个 JSON 字符串数组，内容与你上面 `### ❓` 章节列出的问题 1:1 完全一致（无问题则输出 `[]`）。例如：
===QUESTIONS_JSON===
["1. 关于荷兰角：...", "2. 关于前景遮挡：..."]"""


# ==============================================================================
# 图像编码、遥测与自动切片辅助
# ==============================================================================
def _imread_bgr(path: str) -> np.ndarray | None:
    if HAS_ALBUM and is_raw_file(path):
        qimg = load_photo_qimage(path)
        if not qimg.isNull():
            rgb = qimg.convertToFormat(QImage.Format_RGB888)
            w, h, bpl = rgb.width(), rgb.height(), rgb.bytesPerLine()
            arr = np.frombuffer(rgb.bits(), dtype=np.uint8, count=bpl * h).reshape(h, bpl)[:, : w * 3].reshape(h, w, 3)
            return cv2.cvtColor(np.array(arr), cv2.COLOR_RGB2BGR)
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def _encode_bgr_b64(bgr: np.ndarray, max_dim: int | None = None, quality: int = 90) -> str:
    h, w = bgr.shape[:2]
    out = bgr
    if max_dim and max(w, h) > max_dim:
        scale = max_dim / float(max(w, h))
        out = cv2.resize(bgr, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _extract_json_obj(text: str) -> dict:
    s = text.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 用 raw_decode 精准截断提取完整 JSON 对象，无视尾部多余字符或第二个代码块
    decoder = json.JSONDecoder()
    candidates = []
    pos = 0
    while True:
        idx = s.find("{", pos)
        if idx < 0:
            break
        try:
            obj, end_idx = decoder.raw_decode(s, idx)
            if isinstance(obj, dict):
                candidates.append(obj)
            pos = end_idx
        except Exception:
            pos = idx + 1

    if candidates:
        for c in reversed(candidates):
            if "step_id" in c or "visual_evidence" in c or "status" in c:
                return c
        return candidates[-1]

    raise ValueError("模型未返回合法 JSON 对象")


def _build_telemetry_and_crops(photo_path: str, work_dir: Path, ref_path: str = "",
                               character_name: str = "", extra_ctx: dict = None) -> dict:
    crops_dir = work_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    bgr = _imread_bgr(photo_path)
    if bgr is None:
        raise RuntimeError(f"无法读取照片像素: {photo_path}")

    orig_h, orig_w = bgr.shape[:2]

    # 1. EXIF & AF 遥测
    exif = parse_photo_exif(photo_path) if HAS_ALBUM else {}
    af_boxes = extract_af_points(photo_path, 1) if HAS_ALBUM else []

    # 直方图硬指标（死白/死黑/色彩通道方差）
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    total_px = float(orig_w * orig_h)
    highlight_clip_pct = round(float(np.sum(gray >= 253)) / total_px * 100.0, 2)
    shadow_clip_pct = round(float(np.sum(gray <= 5)) / total_px * 100.0, 2)
    b_ch, g_ch, r_ch = cv2.split(bgr)
    is_monochrome = bool(np.mean(np.abs(r_ch.astype(np.int16) - g_ch.astype(np.int16))) < 3.0 and
                         np.mean(np.abs(g_ch.astype(np.int16) - b_ch.astype(np.int16))) < 3.0)

    # 2. 人脸检测与 1:1 切片
    face_box = None  # (x, y, w, h)
    if HAS_FOCUS_INSPECTOR:
        try:
            scale = 1280.0 / max(orig_w, orig_h)
            gw, gh = int(orig_w * scale), int(orig_h * scale)
            pad_w, pad_h = (gw + 31) // 32 * 32, (gh + 31) // 32 * 32
            pad_img = np.zeros((pad_h, pad_w, 3), dtype=np.uint8)
            pad_img[:gh, :gw] = cv2.resize(bgr, (gw, gh))
            faces = FocusInspector._detect_faces_on_chip(pad_img)
            if faces:
                faces.sort(key=lambda f: f["box"][2] * f["box"][3], reverse=True)
                bx, by, bw, bh = faces[0]["box"]
                face_box = (int(bx / scale), int(by / scale), int(bw / scale), int(bh / scale))
        except Exception:
            face_box = None

    if face_box is None:
        # 兜底：中心偏上区域
        fw, fh = int(orig_w * 0.25), int(orig_h * 0.25)
        face_box = ((orig_w - fw) // 2, int(orig_h * 0.15), fw, fh)

    fx, fy, fw, fh = face_box
    face_area_pct = round((fw * fh) / total_px * 100.0, 2)

    # 脸 + 脖子 1:1 切片（向下扩 50% 吃进锁骨与脖子，左右扩 20%）
    fn_x1 = max(0, int(fx - fw * 0.20))
    fn_x2 = min(orig_w, int(fx + fw * 1.20))
    fn_y1 = max(0, int(fy - fh * 0.15))
    fn_y2 = min(orig_h, int(fy + fh * 1.55))
    crop_face_neck = bgr[fn_y1:fn_y2, fn_x1:fn_x2]

    # 发际线/头顶 1:1 切片
    hl_x1 = max(0, int(fx - fw * 0.30))
    hl_x2 = min(orig_w, int(fx + fw * 1.30))
    hl_y1 = max(0, int(fy - fh * 0.55))
    hl_y2 = min(orig_h, int(fy + fh * 0.45))
    crop_hairline = bgr[hl_y1:hl_y2, hl_x1:hl_x2]

    # 躯干/半身切片（以人脸为中心向下展开）
    bd_x1 = max(0, int(fx - fw * 1.4))
    bd_x2 = min(orig_w, int(fx + fw * 2.4))
    bd_y1 = max(0, int(fy - fh * 0.2))
    bd_y2 = min(orig_h, int(fy + fh * 4.5))
    crop_body = bgr[bd_y1:bd_y2, bd_x1:bd_x2]

    # 眼部/面部拉普拉斯锐度
    face_roi = gray[max(0, fy):min(orig_h, fy + fh), max(0, fx):min(orig_w, fx + fw)]
    face_lap_var = round(float(cv2.Laplacian(face_roi, cv2.CV_64F).var()), 1) if face_roi.size > 0 else 0.0

    # 保存物理切片文件供可视化查看
    p_full = crops_dir / "full.jpg"
    p_fn = crops_dir / "face_neck_1to1.jpg"
    p_hl = crops_dir / "hairline_1to1.jpg"
    p_bd = crops_dir / "body.jpg"

    # 缩放保存全景图（长边 1600px）
    scale_full = min(1.0, 1600.0 / max(orig_w, orig_h))
    bgr_full = cv2.resize(bgr, (int(orig_w * scale_full), int(orig_h * scale_full)), interpolation=cv2.INTER_AREA) if scale_full < 1.0 else bgr

    for path_out, mat in ((p_full, bgr_full), (p_fn, crop_face_neck), (p_hl, crop_hairline), (p_bd, crop_body)):
        ok, buf = cv2.imencode(".jpg", mat, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if ok:
            path_out.write_bytes(buf.tobytes())

    b64_map = {
        "full": _encode_bgr_b64(bgr_full, max_dim=1600, quality=88),
        "face_neck": _encode_bgr_b64(crop_face_neck, max_dim=1280, quality=92),
        "hairline": _encode_bgr_b64(crop_hairline, max_dim=1280, quality=92),
        "body": _encode_bgr_b64(crop_body, max_dim=1400, quality=90),
    }

    if ref_path and os.path.exists(ref_path):
        ref_bgr = _imread_bgr(ref_path)
        if ref_bgr is not None:
            b64_map["ref"] = _encode_bgr_b64(ref_bgr, max_dim=1024, quality=88)

    telemetry = {
        "resolution": f"{orig_w}x{orig_h}",
        "exif": exif,
        "af_points_count": len(af_boxes),
        "face_bbox_px": {"x": fx, "y": fy, "w": fw, "h": fh},
        "face_area_pct": face_area_pct,
        "face_laplacian_sharpness": face_lap_var,
        "highlight_clip_pct": highlight_clip_pct,
        "shadow_clip_pct": shadow_clip_pct,
        "is_monochrome_bw": is_monochrome,
        "character_input": character_name or "未指定（请结合画面自行识别或标明未知）",
        "shooting_context": extra_ctx or {},
    }
    (work_dir / "meta.json").write_text(json.dumps(telemetry, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "telemetry": telemetry,
        "b64_map": b64_map,
        "crop_paths": {
            "full": str(p_full),
            "face_neck": str(p_fn),
            "hairline": str(p_hl),
            "body": str(p_bd),
        }
    }


def _call_vlm_api(base_url: str, api_key: str, model: str, system_prompt: str,
                  user_text: str, image_b64_list: list[tuple[str, str]], timeout: int = 90) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    content_parts = []
    for label, b64 in image_b64_list:
        if not b64:
            continue
        content_parts.append({"type": "text", "text": f"[附带图像：{label}]"})
        content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    content_parts.append({"type": "text", "text": user_text})

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_parts}
        ],
        "temperature": 0.2
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        },
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {e.code}: {err_body[:400]}")


# ==============================================================================
# 后台状态机工作线程 (Pass 1 -> 校验 -> Pass 2 -> Pass 3)
# ==============================================================================
class AuditPipelineWorker(QThread):
    sig_log = Signal(str, str)
    sig_crops_ready = Signal(dict, dict)       # crop_paths, telemetry
    sig_step_updated = Signal(int, int, dict)  # step_id, status(0/1/2), json_obj
    sig_report_ready = Signal(str, list)       # report_md, ambiguous_questions
    sig_finished = Signal(bool, str)

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg

    def _run_custom_python_hook(self, py_path: str, ctx: dict) -> dict:
        if not py_path or not os.path.exists(py_path):
            return {}
        try:
            spec = importlib.util.spec_from_file_location("anicrop_custom_hook", py_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "run_hook") and callable(mod.run_hook):
                res = mod.run_hook(ctx)
                return res if isinstance(res, dict) else {"result": str(res)}
            return {"info": "已加载脚本，但未找到 run_hook(context) 函数"}
        except Exception as e:
            return {"hook_error": str(e)}

    def run(self):
        try:
            photo_path = self.cfg["photo_path"]
            work_dir = Path(self.cfg["work_dir"])
            work_dir.mkdir(parents=True, exist_ok=True)

            self.sig_log.emit("🔬 正在提取 EXIF 遥测数据并生成 1:1 多尺度 ROI 切片...", "INFO")
            prep = _build_telemetry_and_crops(
                photo_path, work_dir,
                ref_path=self.cfg.get("ref_path", ""),
                character_name=self.cfg.get("character_name", ""),
                extra_ctx=self.cfg.get("shooting_context", {})
            )
            telemetry = prep["telemetry"]
            b64_map = prep["b64_map"]
            self.sig_crops_ready.emit(prep["crop_paths"], telemetry)

            telemetry_str = json.dumps(telemetry, ensure_ascii=False)
            char_str = telemetry["character_input"]
            sys_prompt = self.cfg["system_prompt"]
            base_url = self.cfg["base_url"]
            api_key = self.cfg["api_key"]
            model = self.cfg["model"]
            max_workers = self.cfg.get("max_workers", 4)
            steps_cfg = self.cfg["steps"]  # [(id, name, img_keys, instruction), ...]

            img_name_cn = {
                "full": "全景图",
                "face_neck": "1:1 脸脖切片",
                "hairline": "1:1 发际线切片",
                "body": "躯干/动作切片",
                "ref": "官方立绘参考图"
            }

            # ---------------- Pass 1: 并发带图初检 (status: 0 -> 1) ----------------
            self.sig_log.emit("⚡ [Pass 1] 启动 9 步并发带图审计...", "INFO")
            pass1_results = {}

            def _exec_pass1(step_item):
                sid, sname, img_keys, s_inst = step_item
                imgs = [(img_name_cn.get(k, k), b64_map[k]) for k in img_keys if k in b64_map]
                if not imgs:
                    imgs = [("全景图", b64_map["full"])]
                desc = "、".join(lbl for lbl, _ in imgs)
                user_txt = (
                    f"【当前任务】：执行 Pass 1 —— 第 {sid} 步（{sname}）专项看图检查。\n"
                    f"输出纯 JSON，包含字段：step_id({sid}), step_name(\"{sname}\"), status(1 或 未入镜时直接设为2), "
                    f"applicable(bool), visual_evidence(list), strengths(list), flaws(list of {{issue, severity, fix_hint}}), "
                    f"ambiguous_for_user(list of {{observation, question_for_user}}), pass2_cross_check_notes(\"\")。\n\n"
                    f"【物理遥测数据】：{telemetry_str}\n"
                    f"【角色设定】：{char_str}\n"

                    f"【已挂载图片】：{desc}\n\n"
                    f"【本步检查指令】：\n{s_inst}"
                )
                raw = _call_vlm_api(base_url, api_key, model, sys_prompt, user_txt, imgs)
                obj = _extract_json_obj(raw)
                obj["step_id"] = sid
                obj["step_name"] = sname
                obj["status"] = 1
                (work_dir / f"step_{sid}.json").write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
                return sid, obj

            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futs = {pool.submit(_exec_pass1, item): item[0] for item in steps_cfg}
                for fut in as_completed(futs):
                    sid = futs[fut]
                    try:
                        _, obj = fut.result()
                        pass1_results[sid] = obj
                        self.sig_step_updated.emit(sid, obj["status"], obj)
                        self.sig_log.emit(f"✅ [Pass 1] Step {sid} 完成 (status={obj['status']})", "SUCCESS")
                    except Exception as e:
                        self.sig_log.emit(f"❌ [Pass 1] Step {sid} 失败: {e}", "ERROR")
                        raise RuntimeError(f"Pass 1 Step {sid} 异常: {e}")

            # 门禁校验：必须 9 步全部 >= 1
            if len(pass1_results) < len(steps_cfg):
                raise RuntimeError("Pass 1 文件未齐全，终止进入 Pass 2")

            # ---------------- Pass 2: 带图二遍交叉核验 (status: 1 -> 2) ----------------
            self.sig_log.emit("🔍 [Pass 2] 门禁通过！开始带图二遍交叉核验（status 1 -> 2）...", "INFO")
            all_p1_str = json.dumps([pass1_results[s[0]] for s in steps_cfg], indent=2, ensure_ascii=False)
            pass2_tpl = self.cfg["pass2_prompt"]
            pass2_results = dict(pass1_results)

            def _exec_pass2(step_item):
                sid, sname, img_keys, _ = step_item
                if pass1_results[sid].get("status") == 2 and pass1_results[sid].get("applicable") is False:
                    return sid, pass1_results[sid]
                # Pass 2 强制带全景图 + 本步专属切片，彻底杜绝脱图瞎编
                keys_p2 = list(dict.fromkeys(["full"] + img_keys))
                imgs = [(img_name_cn.get(k, k), b64_map[k]) for k in keys_p2 if k in b64_map]
                user_txt = (
                    pass2_tpl.replace("{step_id}", str(sid))
                             .replace("{step_name}", sname)
                             .replace("{EXIF_TELEMETRY}", telemetry_str)
                             .replace("{ALL_9_STEP_JSONS}", all_p1_str)
                )
                raw = _call_vlm_api(base_url, api_key, model, sys_prompt, user_txt, imgs)
                obj = _extract_json_obj(raw)
                obj["step_id"] = sid
                obj["step_name"] = sname
                obj["status"] = 2
                (work_dir / f"step_{sid}_pass2.json").write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
                return sid, obj

            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futs = {pool.submit(_exec_pass2, item): item[0] for item in steps_cfg}
                for fut in as_completed(futs):
                    sid = futs[fut]
                    try:
                        _, obj = fut.result()
                        pass2_results[sid] = obj
                        self.sig_step_updated.emit(sid, 2, obj)
                        self.sig_log.emit(f"🛡️ [Pass 2] Step {sid} 核验通过 (status=2)", "SUCCESS")
                    except Exception as e:
                        self.sig_log.emit(f"❌ [Pass 2] Step {sid} 失败: {e}", "ERROR")
                        raise RuntimeError(f"Pass 2 Step {sid} 异常: {e}")

            # ---------------- Pass 3: 全员 status=2 汇总终审报告 ----------------
            self.sig_log.emit("📝 [Pass 3] 所有文件 status=2 达标！正在生成终审报告与存疑反问...", "INFO")
            all_p2_str = json.dumps([pass2_results[s[0]] for s in steps_cfg], indent=2, ensure_ascii=False)
            pass3_txt = (
                self.cfg["pass3_prompt"]
                    .replace("{EXIF_TELEMETRY}", telemetry_str)
                    .replace("{CHARACTER_INFO}", char_str)
                    .replace("{ALL_9_VERIFIED_JSONS}", all_p2_str)
            )
            imgs_p3 = [(img_name_cn[k], b64_map[k]) for k in ("full", "face_neck", "ref") if k in b64_map]
            # 覆盖 System 的纯 JSON 限制，让 Pass 3 稳稳输出 Markdown + ===QUESTIONS_JSON===
            p3_sys = sys_prompt + "\n\n【格式覆盖】：本次输出以 Markdown 报告为主体，末尾附带 ===QUESTIONS_JSON=== 数组，不要用外层大 JSON 包裹全文。"
            raw_p3 = _call_vlm_api(base_url, api_key, model, p3_sys, pass3_txt, imgs_p3)

            synced_questions = []
            report_md = raw_p3
            if "===QUESTIONS_JSON===" in raw_p3:
                parts = raw_p3.split("===QUESTIONS_JSON===", 1)
                report_md = parts[0].strip()
                q_raw = parts[1].strip()
                if q_raw.startswith("```"):
                    q_raw = "\n".join(q_raw.splitlines()[1:-1]).strip()
                try:
                    q_list = json.loads(q_raw)
                    if isinstance(q_list, list):
                        synced_questions = [{"question_for_user": str(q)} for q in q_list[:6] if str(q).strip()]
                except Exception:
                    synced_questions = []

            # 兜底：若模型没按分隔符输出，从报告的 ❓ 章节提取编号行，保证问卷与报告 1:1 同步
            if not synced_questions and "❓" in report_md:
                tail = report_md.split("❓", 1)[1]
                for line in tail.splitlines()[1:]:
                    ln = line.strip()
                    if ln and (ln[0].isdigit() or ln.startswith(("-", "*", "•"))):
                        synced_questions.append({"question_for_user": ln.lstrip("-*• ")})
                    if len(synced_questions) >= 5:
                        break

            (work_dir / "final_report.md").write_text(report_md, encoding="utf-8")
            self.sig_report_ready.emit(report_md, synced_questions)
            self.sig_finished.emit(True, str(work_dir))

        except Exception as e:
            self.sig_finished.emit(False, str(e))


class FollowUpWorker(QThread):
    sig_done = Signal(bool, str)

    def __init__(self, cfg: dict, report_md: str, user_payload: dict, b64_full: str, telemetry: dict = None):
        super().__init__()
        self.cfg = cfg
        self.report_md = report_md
        self.user_payload = user_payload
        self.b64_full = b64_full
        self.telemetry = telemetry or {}

    def run(self):
        try:
            # 覆盖全局 System Prompt 的 JSON 强约束，彻底杜绝返回 {"audit_report": ...}
            calib_sys = (
                self.cfg["system_prompt"]
                + "\n\n【重要格式覆盖】：本轮为终审校准对话，严禁输出 JSON 格式或 ```json 代码块！必须直接输出纯净的 Markdown 文本。"
            )

            ctrl_model = self.user_payload.get("model_controllable", True)
            ctrl_light = self.user_payload.get("light_controllable", True)
            qa_items = self.user_payload.get("qa_items", [])
            misread_note = self.user_payload.get("misread_note", "").strip()
            telemetry_str = json.dumps(self.telemetry, ensure_ascii=False) if self.telemetry else "未读取到完整 EXIF"

            qa_lines = []
            for idx, item in enumerate(qa_items, 1):
                qa_lines.append(
                    f"{idx}. 问题：{item['question']}\n"
                    f"   - 用户状态判定：【{item['status']}】\n"
                    f"   - 用户补充：{item['note'] or '无'}"
                )
            qa_block = "\n".join(qa_lines) if qa_lines else "（无特定存疑项）"

            speech_rule = (
                "若模特可沟通，必须在抢救方案之后单独输出 `### 🗣️ 现场跟老师沟通的话术`，用高情商、自然口语化的表达（不要甩干瘪摄影术语），教摄影师怎么直接开口让 Coser 配合微调。"
                if ctrl_model else
                "由于本次模特不可控，跳过现场沟通话术章节，不要输出重拍/摆姿建议。"
            )

            user_txt = (
                f"【上一轮审片报告】：\n{self.report_md}\n\n"
                f"【原始器材与物理遥测数据】：\n{telemetry_str}\n\n"
                f"【现场可控权限声明】：\n"
                f"- 老师/模特能否现场沟通摆拍：{'可以沟通指导' if ctrl_model else '不可控（蹭拍/路人抓拍/无权限指挥）'}\n"
                f"- 现场主光是否为摄影师自己打的：{'是（自己布光）' if ctrl_light else '否（现场环境光/蹭别人灯，灯光不可控）'}\n\n"
                f"【摄影师问卷逐条反馈】：\n{qa_block}\n\n"
                f"【摄影师指出 AI 理解错的地方 / 额外说明】：\n{misread_note or '无'}\n\n"
                f"【终审校准铁律（必须严格执行）】：\n"
                f"1. **绝不强行洗白与轻信辩解（硬骨头铁律）**：\n"
                f"   - 只有当用户明确指出【AI 事实理解错了】或给出极具说服力的客观设定/叙事依据时，才撤回对应事实指控；\n"
                f"   - 若用户选择【刻意设计/我觉得这样好看】或给出的理由没有绝对说服力：必须明确表示“尊重你的个人审美选择”，但绝不替用户强行洗白，直接表态“尊重但保留原判意见”，该缺陷依然保留在【修正后的缺点】中。\n"
                f"2. **不可控场景零废话**：若声明模特不可控，严禁给出“让模特换姿势/重拍”的建议；若声明灯光不可控，严禁给出“调整灯位/加轮廓灯”的建议。\n"
                f"3. {speech_rule}\n\n"
                f"【强制输出结构（严格按此 Markdown 章节顺序输出）】：\n"
                f"### 🎯 终审校准定性\n"
                f"### 🔍 问卷反馈逐条裁决\n"
                f"### ✅ 修正后的亮点\n（结合用户确认的真实设定与现场条件，重新梳理真正成立的亮点）\n"
                f"### ⚠️ 修正后的缺点\n（剔除已证实是 AI 误判的项，保留所有客观成立及“尊重但保留意见”的缺点，分级列出）\n"
                f"### 🛠️ 后期抢救与重构指南\n"
                f"{'### 🗣️ 现场跟老师沟通的话术' + chr(10) if ctrl_model else ''}"
                f"### 📊 照片的总体评价（结合器材看）\n"
                f"请严格拆成以下 5 个小点做最终总结：\n"
                f"1. **结合器材看**：结合 EXIF 机身、镜头型号、焦段、光圈、快门、ISO 与摄影师资历，客观评价这套硬件在当前条件下的发挥与物理极限（有没有喂饱高像素、安全快门与景深控制如何）。"
                f"【动态反差表达】：严禁套用固定台词，必须针对实际器材与实际画面的反差现场发挥——若用入门机/高性价比头榨出了惊艳画质，给出真实且具体的技术赞叹；若用顶级旗舰器材却拍出低级硬伤，先针对具体失误锐评吐槽一句再注明“开个玩笑，说正经的：”并切回专业复盘；正常发挥则客观平实总结。\n"
                f"2. **照片本身总结**：作为一张独立成片的综合完成度、视觉冲击力与最终可交付价值。\n"
                f"3. **摄影师技术总结**：结合可控/不可控条件，评价摄影师本人的机位嗅觉、曝光把控、时机抓取与后期审美取舍。\n"
                f"4. **Coser 老师总结**：评价老师的角色还原度、妆造假发准备、镜头表现力与肢体情绪状态。\n"
                f"5. **场景总结**：评价拍摄场地/展台的布景美观度、灯光杂乱度、空间干扰程度以及对出片上限的影响。"
            )

            res = _call_vlm_api(
                self.cfg["base_url"], self.cfg["api_key"], self.cfg["model"],
                calib_sys, user_txt, [("全景图", self.b64_full)]
            )
            if isinstance(res, tuple):
                res = res[0]

            # 兜底防线：万一模型还是手滑吐了 JSON，自动拆包取出正文
            s = res.strip()
            if s.startswith("{") or s.startswith("```json"):
                try:
                    obj = _extract_json_obj(s)
                    for k in ("audit_report", "report", "content", "markdown", "conclusion"):
                        if isinstance(obj.get(k), str) and obj[k].strip():
                            res = obj[k]
                            break
                except Exception:
                    pass

            self.sig_done.emit(True, res)
        except Exception as e:
            self.sig_done.emit(False, str(e))


# ==============================================================================
# 可内嵌主工作台组件: VisionAuditPageWidget
# ==============================================================================
class VisionAuditPageWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.st = QSettings("AnicropGeeks", "AnicropStudio")
        self.photo_path = ""
        self.ref_path = ""
        self.custom_py_path = ""
        self.step_jsons = {}
        self.current_work_dir = ""
        self.last_report_md = ""
        self.last_b64_full = ""
        self.last_telemetry = {}
        self.worker = None
        self.follow_worker = None

        self._init_ui()
        self._load_saved_settings()

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # 1. 顶部 API 与模型配置栏
        api_bar = QFrame()
        api_bar.setStyleSheet("QFrame { background-color: #141418; border: 1px solid #27272f; border-radius: 6px; padding: 6px; }")
        ab_l = QHBoxLayout(api_bar)
        ab_l.setContentsMargins(8, 4, 8, 4)
        ab_l.setSpacing(8)

        ab_l.addWidget(QLabel("API Base URL:"))
        self.edit_base_url = QLineEdit("https://api.deepseek.com/v1")
        self.edit_base_url.setMinimumWidth(210)
        ab_l.addWidget(self.edit_base_url)

        ab_l.addWidget(QLabel("Model:"))
        self.edit_model = QLineEdit("deepseek-flash")
        self.edit_model.setFixedWidth(150)
        ab_l.addWidget(self.edit_model)

        ab_l.addWidget(QLabel("API Key:"))
        self.edit_api_key = QLineEdit()
        self.edit_api_key.setEchoMode(QLineEdit.Password)
        self.edit_api_key.setPlaceholderText("sk-...")
        self.edit_api_key.setMinimumWidth(160)
        ab_l.addWidget(self.edit_api_key)

        ab_l.addWidget(QLabel("并发:"))
        self.spin_workers = QSpinBox()
        self.spin_workers.setRange(1, 9)
        self.spin_workers.setValue(4)
        self.spin_workers.setFixedWidth(55)
        ab_l.addWidget(self.spin_workers)

        self.btn_run = QPushButton("🚀 启动 9 步双遍审计")
        self.btn_run.setStyleSheet(
            "QPushButton { background-color: #0284c7; color: white; font-weight: bold; padding: 6px 16px; border-radius: 4px; border: none; }"
            "QPushButton:hover { background-color: #0369a1; }"
            "QPushButton:disabled { background-color: #27272a; color: #71717a; }"
        )
        self.btn_run.clicked.connect(self._start_audit)
        ab_l.addWidget(self.btn_run)

        root.addWidget(api_bar)

        # 2. 主体三栏分割器 (左：放图片+放Python | 中：放提示词 | 右：状态机+报告+反问)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setStyleSheet("QSplitter::handle { background-color: #27272f; width: 2px; }")

        # ---------------- 左栏：图片、参考图、外挂 Python 文件、切片预览 ----------------
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        left_box = QWidget()
        ll = QVBoxLayout(left_box)
        ll.setContentsMargins(0, 0, 6, 0)
        ll.setSpacing(10)

        # 1) 放主图
        grp_img = QGroupBox("🖼️ 1. 待审照片（支持直接拖拽）")
        gi_l = QVBoxLayout(grp_img)
        self.lbl_drop_photo = QLabel("将漫展原片拖拽到这里\n或点击下方浏览")
        self.lbl_drop_photo.setAcceptDrops(True)
        self.lbl_drop_photo.setAlignment(Qt.AlignCenter)
        self.lbl_drop_photo.setMinimumHeight(150)
        self.lbl_drop_photo.setStyleSheet("QLabel { background-color: #141418; border: 2px dashed #3f3f46; border-radius: 6px; color: #71717a; }")
        self.lbl_drop_photo.dragEnterEvent = lambda e: e.acceptProposedAction() if e.mimeData().hasUrls() else None
        self.lbl_drop_photo.dropEvent = self._on_photo_dropped
        gi_l.addWidget(self.lbl_drop_photo)

        h_img_btn = QHBoxLayout()
        btn_pick_photo = QPushButton("📂 选择照片...")
        btn_pick_photo.clicked.connect(self._browse_photo)
        h_img_btn.addWidget(btn_pick_photo)
        self.edit_char_name = QLineEdit()
        self.edit_char_name.setPlaceholderText("角色名/作品（可选，如：流萤）")
        h_img_btn.addWidget(self.edit_char_name, stretch=1)
        gi_l.addLayout(h_img_btn)

        h_ref = QHBoxLayout()
        self.edit_ref_path = QLineEdit()
        self.edit_ref_path.setPlaceholderText("可选：官方立绘参考图路径...")
        h_ref.addWidget(self.edit_ref_path)
        btn_pick_ref = QPushButton("🎨 立绘")
        btn_pick_ref.clicked.connect(self._browse_ref)
        h_ref.addWidget(btn_pick_ref)
        gi_l.addLayout(h_ref)
        ll.addWidget(grp_img)

        # 2) 拍摄现场与人员背景（先验上下文 + 可控权限）
        grp_ctx = QGroupBox("📋 2. 拍摄现场与人员背景（先验注入）")
        gc_ctx = QVBoxLayout(grp_ctx)
        gc_ctx.setSpacing(6)

        h_expo = QHBoxLayout()
        h_expo.addWidget(QLabel("展会名称:"))
        self.edit_expo_name = QLineEdit()
        self.edit_expo_name.setPlaceholderText("可选，如：碧蓝清凉节 / BW2026")
        h_expo.addWidget(self.edit_expo_name, stretch=1)
        gc_ctx.addLayout(h_expo)

        h_cn = QHBoxLayout()
        h_cn.addWidget(QLabel("老师 CN:"))
        self.edit_coser_cn = QLineEdit()
        self.edit_coser_cn.setPlaceholderText("可选，知名老师可填 CN")
        h_cn.addWidget(self.edit_coser_cn, stretch=1)
        gc_ctx.addLayout(h_cn)

        h_loc = QHBoxLayout()
        h_loc.addWidget(QLabel("拍摄位置:"))
        self.combo_scene_loc = QComboBox()
        self.combo_scene_loc.addItems([
            "自动识别（未指定）", "官方展台", "官方集邮点", "漫展摄影区",
            "漫展公共通道", "专业棚拍", "室内实景", "户外外景", "其他"
        ])
        self.combo_scene_loc.currentTextChanged.connect(self._on_scene_loc_changed)
        h_loc.addWidget(self.combo_scene_loc, stretch=1)
        gc_ctx.addLayout(h_loc)

        self.edit_scene_other = QLineEdit()
        self.edit_scene_other.setPlaceholderText("请输入具体拍摄场景（如：商场中庭/废墟外景）...")
        self.edit_scene_other.setVisible(False)
        gc_ctx.addWidget(self.edit_scene_other)

        h_exp = QHBoxLayout()
        h_exp.addWidget(QLabel("摄影资历:"))
        self.combo_exp_level = QComboBox()
        self.combo_exp_level.addItems([
            "新手入门（<1年）", "进阶玩家（1~3年）",
            "熟练老手（3~5年）", "资深/商拍（5年+）"
        ])
        h_exp.addWidget(self.combo_exp_level, stretch=1)
        gc_ctx.addLayout(h_exp)

        h_perm = QHBoxLayout()
        self.cb_ctrl_model = QCheckBox("老师可沟通配合")
        self.cb_ctrl_model.setChecked(True)
        self.cb_ctrl_light = QCheckBox("灯光是自己打的")
        self.cb_ctrl_light.setChecked(True)
        h_perm.addWidget(self.cb_ctrl_model)
        h_perm.addWidget(self.cb_ctrl_light)
        h_perm.addStretch()
        gc_ctx.addLayout(h_perm)

        ll.addWidget(grp_ctx)

        # 3) 自动切片预览（看模型到底看到了什么）
        grp_crops = QGroupBox("✂️ 3. 自动生成 1:1 切片预览（喂给模型的视野）")
        gc_l = QHBoxLayout(grp_crops)
        self.crop_labels = {}
        for k, title in (("face_neck", "脸+脖 1:1"), ("hairline", "发际线 1:1"), ("body", "躯干")):
            box = QVBoxLayout()
            img_l = QLabel("待生成")
            img_l.setFixedSize(96, 96)
            img_l.setAlignment(Qt.AlignCenter)
            img_l.setStyleSheet("background-color: #09090b; border: 1px solid #27272f; color: #52525b; font-size: 10px;")
            txt_l = QLabel(title)
            txt_l.setAlignment(Qt.AlignCenter)
            txt_l.setStyleSheet("color: #a1a1aa; font-size: 10px;")
            box.addWidget(img_l)
            box.addWidget(txt_l)
            gc_l.addLayout(box)
            self.crop_labels[k] = img_l
        ll.addWidget(grp_crops)
        ll.addStretch()

        left_scroll.setWidget(left_box)
        splitter.addWidget(left_scroll)

        # ---------------- 中栏：提示词可视化编辑区 ----------------
        mid_box = QGroupBox("📝 4. 提示词工作区（可随时微调并存成预设）")
        ml = QVBoxLayout(mid_box)

        h_pt = QHBoxLayout()
        btn_save_prompts = QPushButton("💾 导出提示词预设")
        btn_save_prompts.clicked.connect(self._export_prompts)
        h_pt.addWidget(btn_save_prompts)
        btn_load_prompts = QPushButton("📥 导入提示词预设")
        btn_load_prompts.clicked.connect(self._import_prompts)
        h_pt.addWidget(btn_load_prompts)
        btn_reset_prompts = QPushButton("↩️ 恢复默认")
        btn_reset_prompts.clicked.connect(self._reset_prompts)
        h_pt.addWidget(btn_reset_prompts)
        h_pt.addStretch()
        ml.addLayout(h_pt)

        self.prompt_tabs = QTabWidget()

        self.txt_sys_prompt = QPlainTextEdit(DEFAULT_SYSTEM_PROMPT)
        self.prompt_tabs.addTab(self.txt_sys_prompt, "全局 System")

        # 9 步独立编辑切换
        step_tab = QWidget()
        st_l = QVBoxLayout(step_tab)
        h_st = QHBoxLayout()
        h_st.addWidget(QLabel("选择步骤:"))
        self.combo_step_edit = QComboBox()
        for sid, sname, _, _ in DEFAULT_STEPS:
            self.combo_step_edit.addItem(f"Step {sid}: {sname}", sid)
        self.combo_step_edit.currentIndexChanged.connect(self._on_step_edit_switched)
        h_st.addWidget(self.combo_step_edit, stretch=1)
        st_l.addLayout(h_st)

        self.step_prompt_texts = {sid: inst for sid, _, _, inst in DEFAULT_STEPS}
        self.txt_step_prompt = QPlainTextEdit(self.step_prompt_texts[1])
        self.txt_step_prompt.textChanged.connect(self._on_step_text_changed)
        st_l.addWidget(self.txt_step_prompt)
        self.prompt_tabs.addTab(step_tab, "Pass 1 (1~9步)")

        self.txt_pass2_prompt = QPlainTextEdit(DEFAULT_PASS2_PROMPT)
        self.prompt_tabs.addTab(self.txt_pass2_prompt, "Pass 2 (交叉核验)")

        self.txt_pass3_prompt = QPlainTextEdit(DEFAULT_PASS3_PROMPT)
        self.prompt_tabs.addTab(self.txt_pass3_prompt, "Pass 3 (终审报告)")

        ml.addWidget(self.prompt_tabs)
        splitter.addWidget(mid_box)

        # ---------------- 右栏：状态机看板 + 报告 + 反问交互 ----------------
        right_box = QGroupBox("📊 5. 状态机实时监控与审片输出")
        rl = QVBoxLayout(right_box)

        # 9 步状态灯条
        grid_status = QHBoxLayout()
        grid_status.setSpacing(4)
        self.step_btns = {}
        for sid, sname, _, _ in DEFAULT_STEPS:
            b = QPushButton(f"S{sid}:0")
            b.setToolTip(f"Step {sid}: {sname}（点击查看该步 JSON 输出）")
            b.setFixedHeight(28)
            b.clicked.connect(lambda ch=False, s=sid: self._inspect_step_json(s))
            grid_status.addWidget(b)
            self.step_btns[sid] = b
        rl.addLayout(grid_status)
        self._reset_step_badges()

        self.out_tabs = QTabWidget()
        self.txt_report = QTextEdit()
        self.txt_report.setReadOnly(True)
        self.txt_report.setStyleSheet("background-color: #09090b; border: 1px solid #27272f; font-size: 12px; color: #e4e4e7; padding: 8px;")
        self.out_tabs.addTab(self.txt_report, "🎯 终审报告")

        self.txt_step_json = QPlainTextEdit()
        self.txt_step_json.setReadOnly(True)
        self.txt_step_json.setStyleSheet("background-color: #09090b; border: 1px solid #27272f; font-family: Consolas; font-size: 11px; color: #86efac;")
        self.out_tabs.addTab(self.txt_step_json, "📄 单步 JSON 详情")

        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setStyleSheet("background-color: #09090b; border: 1px solid #27272f; font-family: Consolas; font-size: 11px; color: #d4d4d8;")
        self.out_tabs.addTab(self.txt_log, "🖥️ 运行日志")

        

        # 摄影师动态调查问卷与校准区
        follow_frame = QFrame()
        follow_frame.setStyleSheet("QFrame { background-color: #18181e; border: 1px solid #27272f; border-radius: 6px; padding: 6px; }")
        ff_l = QVBoxLayout(follow_frame)
        ff_l.setContentsMargins(6, 6, 6, 6)
        ff_l.setSpacing(6)

        ff_l.addWidget(QLabel("📋 存疑问卷（与报告 1:1 同步，勾选确认意图）："))

        self.qa_scroll = QScrollArea()
        self.qa_scroll.setWidgetResizable(True)
        self.qa_scroll.setMinimumHeight(180)
        self.qa_scroll.setStyleSheet("QScrollArea { border: 1px solid #27272f; background: #121216; border-radius: 4px; }")
        self.qa_container = QWidget()
        self.qa_layout = QVBoxLayout(self.qa_container)
        self.qa_layout.setContentsMargins(6, 6, 6, 6)
        self.qa_layout.setSpacing(6)
        self.qa_placeholder = QLabel("跑完 9 步审计后，AI 存疑点将自动生成调查问卷...")
        self.qa_placeholder.setStyleSheet("color: #71717a; font-size: 11px;")
        self.qa_layout.addWidget(self.qa_placeholder)
        self.qa_scroll.setWidget(self.qa_container)
        ff_l.addWidget(self.qa_scroll)
        self.qa_rows = []

        h_f = QHBoxLayout()
        h_f.addWidget(QLabel("AI理解错的地方:"))
        self.edit_followup = QLineEdit()
        self.edit_followup.setPlaceholderText("指出 AI 事实看错的地方（如：角色设定原本就是机械爪/那不是杂物是道具）...")
        h_f.addWidget(self.edit_followup, stretch=1)
        self.btn_followup = QPushButton("🚀 提交问卷校准")
        self.btn_followup.setEnabled(False)
        self.btn_followup.setStyleSheet("QPushButton { background-color: #0284c7; color: white; font-weight: bold; padding: 5px 12px; border: none; } QPushButton:disabled { background-color: #27272a; color: #71717a; }")
        self.btn_followup.clicked.connect(self._send_followup)
        h_f.addWidget(self.btn_followup)
        ff_l.addLayout(h_f)
        right_splitter = QSplitter(Qt.Vertical)
        right_splitter.setStyleSheet("QSplitter::handle { background-color: #27272f; height: 4px; }")
        right_splitter.addWidget(self.out_tabs)
        right_splitter.addWidget(follow_frame)
        right_splitter.setSizes([430, 310])
        rl.addWidget(right_splitter, stretch=1)

        splitter.addWidget(right_box)
        splitter.setSizes([360, 440, 500])
        root.addWidget(splitter, stretch=1)

    def _load_saved_settings(self):
        self.edit_base_url.setText(str(self.st.value("audit_base_url", "https://api.deepseek.com/v1")))
        self.edit_model.setText(str(self.st.value("audit_model", "deepseek-flash")))
        self.edit_api_key.setText(str(self.st.value("audit_api_key", "")))
        self.spin_workers.setValue(int(self.st.value("audit_workers", 4)))
        self.edit_expo_name.setText(str(self.st.value("audit_expo_name", "")))
        saved_exp = str(self.st.value("audit_exp_level", "进阶玩家（1~3年）"))
        idx_exp = self.combo_exp_level.findText(saved_exp)
        if idx_exp >= 0:
            self.combo_exp_level.setCurrentIndex(idx_exp)

    def _save_settings(self):
        self.st.setValue("audit_base_url", self.edit_base_url.text().strip())
        self.st.setValue("audit_model", self.edit_model.text().strip())
        self.st.setValue("audit_api_key", self.edit_api_key.text().strip())
        self.st.setValue("audit_workers", self.spin_workers.value())
        self.st.setValue("audit_expo_name", self.edit_expo_name.text().strip())
        self.st.setValue("audit_exp_level", self.combo_exp_level.currentText())

    def _on_scene_loc_changed(self, text: str):
        self.edit_scene_other.setVisible(text == "其他")
        if text in ("官方集邮点", "漫展公共通道"):
            self.cb_ctrl_model.setChecked(False)
            self.cb_ctrl_light.setChecked(False)
        elif text == "专业棚拍":
            self.cb_ctrl_model.setChecked(True)
            self.cb_ctrl_light.setChecked(True)

    def _reset_step_badges(self):
        for sid, btn in self.step_btns.items():
            btn.setText(f"S{sid}:0")
            btn.setStyleSheet("QPushButton { background-color: #18181e; border: 1px solid #3f3f46; color: #71717a; border-radius: 3px; font-weight: bold; font-size: 11px; }")

    def _on_step_updated(self, sid: int, status: int, obj: dict):
        self.step_jsons[sid] = obj
        btn = self.step_btns.get(sid)
        if not btn:
            return
        btn.setText(f"S{sid}:{status}")
        if status == 1:
            btn.setStyleSheet("QPushButton { background-color: #422006; border: 1px solid #eab308; color: #fde047; border-radius: 3px; font-weight: bold; font-size: 11px; }")
        elif status == 2:
            btn.setStyleSheet("QPushButton { background-color: #052e16; border: 1px solid #22c55e; color: #4ade80; border-radius: 3px; font-weight: bold; font-size: 11px; }")

    def _inspect_step_json(self, sid: int):
        if sid not in self.step_jsons:
            self.txt_step_json.setPlainText(f"Step {sid} 尚未产出数据")
        else:
            self.txt_step_json.setPlainText(json.dumps(self.step_jsons[sid], indent=2, ensure_ascii=False))
        self.out_tabs.setCurrentIndex(1)

    def _on_photo_dropped(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            self.load_photo(urls[0].toLocalFile())
            event.acceptProposedAction()

    def _browse_photo(self):
        f, _ = QFileDialog.getOpenFileName(self, "选择待审照片", "", "图片 (*.jpg *.jpeg *.png *.webp *.cr3 *.cr2 *.arw *.nef *.dng);;所有 (*.*)")
        if f:
            self.load_photo(f)

    def load_photo(self, path: str):
        if not path or not os.path.exists(path):
            return
        self.photo_path = path
        bgr = _imread_bgr(path)
        if bgr is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
            pix = QPixmap.fromImage(qimg).scaled(300, 145, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.lbl_drop_photo.setPixmap(pix)
            self.lbl_drop_photo.setStyleSheet("QLabel { background-color: #141418; border: 2px solid #0284c7; border-radius: 6px; }")
        self._append_log(f"🖼️ 已载入照片: {Path(path).name}", "INFO")

    def _browse_ref(self):
        f, _ = QFileDialog.getOpenFileName(self, "选择官方参考立绘", "", "图片 (*.jpg *.jpeg *.png *.webp);;所有 (*.*)")
        if f:
            self.ref_path = f
            self.edit_ref_path.setText(f)

    def _on_py_dropped(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            p = urls[0].toLocalFile()
            if p.lower().endswith(".py"):
                self._load_py_file(p)
                event.acceptProposedAction()

    def _browse_py(self):
        f, _ = QFileDialog.getOpenFileName(self, "选择自定义 Python 钩子脚本", "", "Python (*.py)")
        if f:
            self._load_py_file(f)

    def _load_py_file(self, path: str):
        self.custom_py_path = path
        self.edit_py_path.setText(path)
        try:
            self.txt_py_preview.setPlainText(Path(path).read_text(encoding="utf-8"))
        except Exception as e:
            self._append_log(f"⚠️ 读取 .py 失败: {e}", "WARN")

    def _save_py_file(self):
        p = self.edit_py_path.text().strip()
        if not p:
            return
        try:
            Path(p).write_text(self.txt_py_preview.toPlainText(), encoding="utf-8")
            self._append_log(f"💾 已保存脚本: {Path(p).name}", "SUCCESS")
        except Exception as e:
            QMessageBox.warning(self, "保存失败", str(e))

    def _on_step_edit_switched(self):
        sid = self.combo_step_edit.currentData()
        self.txt_step_prompt.blockSignals(True)
        self.txt_step_prompt.setPlainText(self.step_prompt_texts.get(sid, ""))
        self.txt_step_prompt.blockSignals(False)

    def _on_step_text_changed(self):
        sid = self.combo_step_edit.currentData()
        self.step_prompt_texts[sid] = self.txt_step_prompt.toPlainText()

    def _export_prompts(self):
        f, _ = QFileDialog.getSaveFileName(self, "导出提示词预设", "audit_prompts.json", "JSON (*.json)")
        if f:
            data = {
                "system": self.txt_sys_prompt.toPlainText(),
                "steps": self.step_prompt_texts,
                "pass2": self.txt_pass2_prompt.toPlainText(),
                "pass3": self.txt_pass3_prompt.toPlainText(),
            }
            Path(f).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def _import_prompts(self):
        f, _ = QFileDialog.getOpenFileName(self, "导入提示词预设", "", "JSON (*.json)")
        if f:
            data = json.loads(Path(f).read_text(encoding="utf-8"))
            self.txt_sys_prompt.setPlainText(data.get("system", DEFAULT_SYSTEM_PROMPT))
            for k, v in (data.get("steps") or {}).items():
                self.step_prompt_texts[int(k)] = v
            self._on_step_edit_switched()
            self.txt_pass2_prompt.setPlainText(data.get("pass2", DEFAULT_PASS2_PROMPT))
            self.txt_pass3_prompt.setPlainText(data.get("pass3", DEFAULT_PASS3_PROMPT))

    def _reset_prompts(self):
        self.txt_sys_prompt.setPlainText(DEFAULT_SYSTEM_PROMPT)
        self.step_prompt_texts = {sid: inst for sid, _, _, inst in DEFAULT_STEPS}
        self._on_step_edit_switched()
        self.txt_pass2_prompt.setPlainText(DEFAULT_PASS2_PROMPT)
        self.txt_pass3_prompt.setPlainText(DEFAULT_PASS3_PROMPT)

    def _append_log(self, text: str, level: str = "INFO"):
        cmap = {"INFO": "#d4d4d8", "SUCCESS": "#4ade80", "WARN": "#facc15", "ERROR": "#f87171"}
        tm = time.strftime("%H:%M:%S")
        self.txt_log.append(f'<span style="color:#71717a;">[{tm}]</span> <span style="color:{cmap.get(level, "#d4d4d8")};">{text}</span>')
        self.txt_log.moveCursor(QTextCursor.End)

    def _start_audit(self):
        if not self.photo_path or not os.path.exists(self.photo_path):
            QMessageBox.warning(self, "缺少照片", "请先拖入或选择一张待审照片！")
            return
        api_key = self.edit_api_key.text().strip()
        if not api_key:
            QMessageBox.warning(self, "缺少 API Key", "请先填入可用的 API Key！")
            return

        self._save_settings()
        self._reset_step_badges()
        self.step_jsons.clear()
        self.txt_report.clear()
        self.btn_run.setEnabled(False)
        self.btn_followup.setEnabled(False)

        cache_base = get_app_cache_root() if HAS_ALBUM else (Path(__file__).resolve().parent / ".cache")
        work_dir = cache_base / "audit_workspace" / f"{Path(self.photo_path).stem}_{int(time.time())}"
        self.current_work_dir = str(work_dir)

        steps_cfg = [(sid, sname, ikeys, self.step_prompt_texts[sid]) for sid, sname, ikeys, _ in DEFAULT_STEPS]
        cfg = {
            "photo_path": self.photo_path,
            "ref_path": self.edit_ref_path.text().strip(),
            "character_name": self.edit_char_name.text().strip(),
            "shooting_context": {
                "expo_name": self.edit_expo_name.text().strip() or "未填写",
                "coser_cn": self.edit_coser_cn.text().strip() or "未填写/未知",
                "scene_location": (self.edit_scene_other.text().strip() or "其他") if self.combo_scene_loc.currentText() == "其他" else self.combo_scene_loc.currentText(),
                "photographer_level": self.combo_exp_level.currentText(),
                "model_controllable": self.cb_ctrl_model.isChecked(),
                "light_controllable": self.cb_ctrl_light.isChecked(),
            },
            "work_dir": str(work_dir),
            "base_url": self.edit_base_url.text().strip(),
            "api_key": api_key,
            "model": self.edit_model.text().strip(),
            "max_workers": self.spin_workers.value(),
            "system_prompt": self.txt_sys_prompt.toPlainText(),
            "steps": steps_cfg,
            "pass2_prompt": self.txt_pass2_prompt.toPlainText(),
            "pass3_prompt": self.txt_pass3_prompt.toPlainText(),
        }

        self.worker = AuditPipelineWorker(cfg)
        self.worker.sig_log.connect(self._append_log)
        self.worker.sig_crops_ready.connect(self._on_crops_ready)
        self.worker.sig_step_updated.connect(self._on_step_updated)
        self.worker.sig_report_ready.connect(self._on_report_ready)
        self.worker.sig_finished.connect(self._on_worker_finished)
        self.worker.start()

    def _on_crops_ready(self, crop_paths: dict, telemetry: dict):
        for k, lbl in self.crop_labels.items():
            p = crop_paths.get(k)
            if p and os.path.exists(p):
                pix = QPixmap(p).scaled(94, 94, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                lbl.setPixmap(pix)
        full_p = crop_paths.get("full")
        self.last_telemetry = telemetry
        self._append_log(f"📐 遥测就绪: 人脸占幅 {telemetry['face_area_pct']}% | 锐度 {telemetry['face_laplacian_sharpness']} | 高光溢出 {telemetry['highlight_clip_pct']}%", "INFO")

    def _rebuild_qa_questionnaire(self, ambiguous_list: list):
        self.qa_rows.clear()
        while self.qa_layout.count():
            item = self.qa_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        if not ambiguous_list:
            lbl = QLabel("✅ 本张照片无明显存疑点，可直接勾选可控权或填写纠正后校准。")
            lbl.setStyleSheet("color: #4ade80; font-size: 11px;")
            self.qa_layout.addWidget(lbl)
            return

        for idx, q_obj in enumerate(ambiguous_list, 1):
            q_text = q_obj.get("question_for_user") or q_obj.get("observation") or str(q_obj)
            row_w = QFrame()
            row_w.setStyleSheet("QFrame { background-color: #18181e; border: 1px solid #27272f; border-radius: 4px; padding: 4px; }")
            rv = QVBoxLayout(row_w)
            rv.setContentsMargins(6, 4, 6, 4)
            rv.setSpacing(4)

            lbl_q = QLabel(f"Q{idx}: {q_text}")
            lbl_q.setWordWrap(True)
            lbl_q.setStyleSheet("color: #e4e4e7; font-size: 11px; border: none;")
            rv.addWidget(lbl_q)

            h_ans = QHBoxLayout()
            combo = QComboBox()
            combo.addItems([
                "确实是失误，接受批评",
                "现场不可控 / 被迫妥协",
                "刻意设计 / 我觉得这样好看",
                "AI 事实看错了（请在右侧/下方纠正）"
            ])
            combo.setMinimumWidth(205)
            edit_note = QLineEdit()
            edit_note.setPlaceholderText("可选补充说明...")
            h_ans.addWidget(combo)
            h_ans.addWidget(edit_note, stretch=1)
            rv.addLayout(h_ans)

            self.qa_layout.addWidget(row_w)
            self.qa_rows.append((q_text, combo, edit_note))

    def _on_report_ready(self, report_md: str, ambiguous_list: list):
        self.last_report_md = report_md
        self.txt_report.setMarkdown(report_md)
        self.out_tabs.setCurrentIndex(0)
        self._rebuild_qa_questionnaire(ambiguous_list)
        self.btn_followup.setEnabled(bool(self.last_report_md))

    def _on_worker_finished(self, ok: bool, msg: str):
        self.btn_run.setEnabled(True)
        if ok:
            self._append_log(f"🎉 全部 9 步双遍审计完成！状态文件已落盘至: {msg}", "SUCCESS")
        else:
            QMessageBox.critical(self, "审计异常", msg)

    def _send_followup(self):
        if not self.last_report_md:
            return
        qa_items = [
            {"question": q_txt, "status": cb.currentText(), "note": ed.text().strip()}
            for q_txt, cb, ed in self.qa_rows
        ]
        user_payload = {
            "model_controllable": self.cb_ctrl_model.isChecked(),
            "light_controllable": self.cb_ctrl_light.isChecked(),
            "qa_items": qa_items,
            "misread_note": self.edit_followup.text().strip(),
        }
        self.btn_followup.setEnabled(False)
        self._append_log("💬 正在按调查问卷与现场可控权校准终审结论...", "INFO")
        cfg = {
            "base_url": self.edit_base_url.text().strip(),
            "api_key": self.edit_api_key.text().strip(),
            "model": self.edit_model.text().strip(),
            "system_prompt": self.txt_sys_prompt.toPlainText(),
        }
        self.follow_worker = FollowUpWorker(cfg, self.last_report_md, user_payload, self.last_b64_full, self.last_telemetry)
        self.follow_worker.sig_done.connect(self._on_followup_done)
        self.follow_worker.start()

    def _on_followup_done(self, ok: bool, res: str):
        self.btn_followup.setEnabled(True)
        if ok:
            self.last_report_md += "\n\n---\n### 🔄 结合摄影师意图校准后的终审结论\n" + res
            self.txt_report.setMarkdown(self.last_report_md)
            self.edit_followup.clear()
            self._append_log("✅ 意图校准完成！", "SUCCESS")
        else:
            self._append_log(f"❌ 校准失败: {res}", "ERROR")


# ==============================================================================
# 独立运行测试入口 (后续随时可作为一个 Tab 嵌进 ui.py)
# ==============================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = VisionAuditPageWidget()
    win.setWindowTitle("Anicrop Studio - 多模态审美审计实验台 (Standalone)")
    win.resize(1380, 860)
    win.setStyleSheet("""
        QWidget { background-color: #0e0e11; color: #e4e4e7; font-family: "Segoe UI", "Microsoft YaHei", sans-serif; font-size: 12px; }
        QGroupBox { background-color: #141418; border: 1px solid #27272f; border-radius: 8px; margin-top: 10px; padding-top: 12px; font-weight: bold; color: #38bdf8; }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
        QLineEdit, QComboBox, QSpinBox, QPlainTextEdit { background-color: #1c1c22; border: 1px solid #3f3f46; border-radius: 4px; padding: 4px 8px; color: #fafafa; }
        QPushButton { background-color: #1f1f24; border: 1px solid #3f3f46; border-radius: 4px; padding: 5px 10px; font-weight: 500; }
        QPushButton:hover { background-color: #2f2f38; }
        QTabWidget::pane { border: 1px solid #27272f; background-color: #141418; border-radius: 4px; }
        QTabBar::tab { background-color: #18181e; color: #a1a1aa; padding: 6px 12px; margin-right: 2px; border-top-left-radius: 4px; border-top-right-radius: 4px; font-weight: bold; }
        QTabBar::tab:selected { background-color: #141418; color: #38bdf8; border-bottom: 2px solid #38bdf8; }
    """)
    win.show()
    sys.exit(app.exec())