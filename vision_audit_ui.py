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
from PySide6.QtGui import (
    QPixmap, QImage, QDragEnterEvent, QDropEvent, QFont, QTextCursor,
    QColorSpace, QImageReader
)

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
4. 【漫展现场物理常识（严禁云评价）】：
   - 【场景纯净度与光质标尺（严禁机械打压）】：漫展 LED 大屏易产生焦外摩尔纹与溢色，严禁无脑吹成顶级。评判场景高低的核心依据是【背景纯净度（路人与杂物干扰程度）】与【现场光质底线（有无恶性死亡顶光/全黑无光）】：只有杂物路人较多明显干扰画面、或光质恶劣时才给【😐 NPC】或【💀 拉完了】；若仅存在少量对主体观感影响不大的边缘杂物或轻微干扰、且无恶性顶光，即可给到【✨ 人上人】；只要背景做到彻底干净（一点杂物、路人穿帮都没有）且光线正常，即可稳拿【💎 顶级】；若在此基础上额外有专业摄影棚级布光或宏大精致置景，直接给到【🔥 夯】；
   - 【景别边界独立准则（严禁拿全身标尺审半身）】：特写、胸像、半身、七分身与全身均是各自独立且完整的合法构图景别。面对明确的半身/胸像/特写作品，**严禁质问“为何未拍到脚部/下肢”或强索全身原片，亦不可强行套用全身站姿重心标准**；仅当构图意图明显为全身或肢体被切不当，比如切在关节处时，才考察肢体截断失误；
   - 【镜头形态与物理空间协同研判（定焦 vs 变焦）】：必须结合 EXIF 中的镜头型号与实际拍摄焦段判断现场构图自由度！若使用**定焦镜头**，或使用**变焦镜头且当前焦段已达到物理最广角端**，在狭窄现场出现取景局促或边缘擦边属于无法再退的“空间物理妥协”；若使用的是**变焦镜头且当前焦段尚有大量更广焦距余量**，则说明当前景别与视角是摄影师**主动做出的焦段构图选择**，严禁盲目套用“退无可退”来强行解释；
   - 【二次元漫展核心交付物是人脸与神态】：合焦精准、高光未崩、神态对味的人像，具有极高的出片与精修交付价值。
5. 【影调骨架、几何投影与色彩分离铁律】：
   - **立体布光必有几何投影**：当采用主光搭配对角轮廓光进行立体布光时，下颌骨在颈部投下方向性阴影、背光侧出现明暗与色温落差属于正常的物理几何现象！严禁将这种立体投影误判为“缺陷/断层”，更严禁建议强行提亮拉平颈部阴影导致面部失去三维体积感，除非这块阴影黑到了完全没有细节且和画面风格不搭；
   - **暗部骨架法则**：严禁把深色服饰、肢体遮挡与背光面的自然阴影判为死黑瑕疵，严禁给出大面积提亮暗部导致画面发灰发平的业余建议；仅当五官神态黑糊不可辨时才记为欠曝；
   - **虚化剥离与防贴纸原则**：主体与背景的剥离不仅依靠明暗与色彩反差，**合焦锐度与焦外虚化的光学反差（Optical Separation）**同样构成强效分离！严禁无视焦外虚化，动辄建议“强行压暗背景”制造不自然的“抠图贴纸感”；
   - **冷暖色温分离即高级感**：干净通透的中性面光搭配贴合主题氛围的暖调或冷调背景，属于高阶的冷暖色温对冲语言。必须尊重现场画面氛围与主题，严禁机械建议“将面光拉得与背景一样暖”或“将背景漂得与面光一样白”来追求虚假的全局色温统一！且漫展环境大多数coser老师偏爱冷白皮，只要面光色温不是特别不协调那都是正常的
6. 【Cosplay 三次元现实工艺与 OOC 审判定律（严禁教条主义二次元洁癖）】：
   - **三次元工艺与物理适配绝非 OOC**：二次元立绘在三次元落地中，Cosplay 服装为了版型合身采用提花、暗纹、浮雕或皮革工艺升级，假发为了防滑脱、透气修饰脸型进行适度长短修剪与发丝打薄，均属于【成立的三次元具象化工艺表达】，绝对严禁教条主义判定为 OOC 或造型扣分！
   - **区分物理环境色偏与道具偏差**：美瞳、假发、布料在现场冷暖强光源透射下产生的正常物理色相漂移（如红瞳在暖光下偏琥珀橙、黑发受强光显棕），属于客观光学规律，严禁当成 OOC 误判；
   - **OOC 的严格红线**：仅当出现核心标志性阵营反转、关键特征彻底遗漏、主色系彻底颠倒、或神态言行严重违背角色灵魂内核时才准判定为 OOC！微小工艺公差一律判定为正向的三次元具象化表达！
7. 【严格 JSON 输出】：只输出合法的纯 JSON 对象，不要包含任何多余闲聊文字。"""

DEFAULT_STEPS = [
(1, "构图与场景", ["full"],
     "1. 【拍摄场景定性】：若遥测明确指定则采信；若为自动识别，依据背景特征客观定性并写明依据。\n"
     "2. 【景别独立性与机位俯仰协调度】：精准判定景别（全身/七分身/半身/胸像/特写），半身与近景无需考察下肢；结合 EXIF 镜头型号（定焦 vs 变焦）与实拍焦段，区分‘定焦/最广端受限’与‘变焦主动构图取舍’；评估机位高低与俯仰角是否与角色体型设定及姿势相协调（区分服务于娇小可爱/特定视角的合理俯仰，与无意识视角导致比例严重失衡、观感不协调的失败透视）；审查三层倾斜滑落感。\n"
     "3. 【九宫格与经典构图法核验（核心力点）】：\n"
     "   - **九宫格与三分法则（Rule of Thirds）**：主视眼、核心面中三角区、标志性武器/道具是否精准锚定在九宫格的黄金力点（交点）或三分线上？还是随意偏离导致画面局促或头顶留白无序？\n"
     "   - **几何构图形式**：画面呈现何种构图几何骨架（中心对称冲击、三角稳定结构、对角线动态张力、引导线构图等）？\n"
     "4. 【重心构图与全画幅视觉天平（Visual Weight）】：\n"
     "   - 审查全画面的视觉重心（Visual Center of Gravity）：综合人物姿态站位、高饱和色块与大面积暗部影调，评估画面左右与上下的视觉平衡感；\n"
     "   - 判断画面是底盘扎实、势均力敌的稳固重心，还是出现严重偏重向一侧滑落、头重脚轻的失衡感？\n"
     "5. 【前中后景三层空间纵深】：前景虚化引导、中景主体光学剥离、背景包覆感与空气感，严查无层次贴片平铺；\n"
     "6. 【切线与刺眼穿帮】：检查四肢死亡截断；高敏扫描现实公共设施、警示标识及杂乱路人。"),
    (2, "角色服化还原度与脸脖光影", ["full", "face_neck"],
     "1. 【三次元现实工艺与角色神韵核验（严禁二次元洁癖）】：\n"
     "   - 对标官方人设检索情报，核验标志性特征（发色体系、核心配件、服装主色调与配饰符号）；\n"
     "   - **破除教条主义**：充分理解商用/定制 Cos 服装的面料工艺提升与假发修剪属于现实人体工学适配，只要核心神韵与标志特征成立，严禁对微小的几厘米发长公差、发尾微卷或服装工艺优化妄断 OOC；严禁将强光透射导致的瞳色/发色物理色偏当成 OOC！\n"
     "2. 【脸脖过渡与布光几何投影鉴别】：盯着 1:1 脸脖切片分析。**牢记布光几何常识：立体主光搭配对角轮廓光必然在颈部产生下颌投影与明暗色温差，这是立体感来源！** 严禁将方向性投影或轮廓透射光晕误判为‘底妆断层’，更严禁建议拉平颈部阴影。仅当脸脖在同角度直射光下出现粉底色号严重脱节（如面死白脖死灰）时才算瑕疵。"),
    (3, "色彩影调与畸变", ["full"],
     "1. 结合 EXIF 曝光三要素，检查面部关键区域是否过曝死白或严重欠曝。坚决区分‘深色衣物/合理投影构成的自然反差’与‘主体欠曝失败’；\n"
     "2. 【主题色温与冷暖分离鉴别】：评估白平衡必须结合现场画面氛围与主题，严格区分‘失控脏色’与‘主动冷暖对冲（如纯净中性面光配暖/冷调氛围背景）’，严禁强求面光与背景色温一致；\n"
     "3. 结合焦段检查镜头拉扯畸变。"),
    (4, "动作与体态", ["full"],
     "1. 动作与角色性格贴合度，动作是否有落点与支撑，有无僵硬假摆。\n"
     "2. 检查体态与透视协调度：不当机位俯仰造成的比例失调、耸肩缩脖、挤出双下巴或重心倾倒。"),
    (5, "人脸与妆造(R)", ["face_neck"],
     "1. 【清晰度与快门安全】：人脸像素与拉普拉斯锐度，眼部是否手抖脱焦。\n"
     "2. 【抓拍瞬态与眼神准则（区分神态翻车与二次元发型风格）】：\n"
     "   - 【生理性翻车（一票否决）】：凡是眼睑半垂无神、视线涣散、翻白眼等抓拍在眨眼中途的，坚决判定为神态失守废片；\n"
     "   - 【二次元发型与侧脸神韵（严禁误判）】：**斜刘海、遮眼发、以及 3/4 侧脸角度下发丝对远视眼的自然遮挡，属于二次元角色极高频的经典视觉语言与人设！只要画面中露出的主视眼合焦刀锋般锐利、眼神情绪对味，必须判定为神态加分项！绝对严禁将角色的标志性斜刘海/发丝误判为‘妆造失误’或‘眼神封印’！**\n"
     "3. 检查卡粉、双眼皮贴反光、假睫毛翘边与妆造细节；厚底妆正常光滑严禁误判磨皮过度。"),
    (6, "头发与发际线", ["hairline"],
     "1. 假发发网是否露馅、真发碎发穿帮、假发头顶压塌与炸毛飞丝。\n"
     "2. 假发发色与造型是否符合角色设定。"),
    (7, "手部专项", ["body"],
     "1. 若未入镜标 applicable=false；\n"
     "2. 检查手部运动模糊、僵硬鸡爪手、手指直戳镜头等体态瑕疵。"),
    (8, "布光公式与色彩节律反推", ["full"],
     "1. 【光路反推与专业控光加分体系（区分大师与普通场照的核心试金石）】：\n"
     "   - **【充盈面光与眼神光（高权重加分项）】**：面部主光充足饱满且光质柔和细腻（呈现柔光箱/反光伞漫反射特征），瞳孔内具有清晰规整的眼神光（Catchlight），五官立体通透而非惨淡平光；\n"
     "   - **【立体轮廓与光学虚化剥离（高权重加分项）】**：综合评估主体与背景的分离手段——既看侧逆光/发丝光对边缘的明暗雕刻，**也必须充分考量景深虚化反差（主体刀锋锐利 vs 背景柔化）带来的天然空气剥离感**！不得忽视光学虚化只看颜色明度；\n"
     "   - **【阴影呼吸感与影调骨架（高权重加分项）】**：光比结构清晰，主光加对角轮廓光自然形成的下颌投影与背光暗部沉稳且有半影过渡（Penumbra），明暗对照（Chiaroscuro）立体有力；\n"
     "2. 【叙事环境光、主题色温与防贴纸辨析】：\n"
     "  - 尊重现场主题与氛围：主光保持纯净通透的肤色还原、背景保留特定冷暖主题色温，属于高阶的冷暖色温分离，严禁机械拉平两者色温；严禁在背景已具备虚化层次时无脑建议大幅压暗背景，以免破坏现场光影逻辑造成“假抠图贴纸感”；\n"
     "3. 【全画幅色彩节律与前后景共振】：\n"
     "   - 严密审查画面中跨越空间层级（虚化前景、主体服饰/配饰/道具、背景主色调）的同色系呼应与冷暖对冲结构。当特定主色调在‘前景—主体—背景’之间形成纵深连续的色彩节律与视线闭环时，必须判定为核心视觉与构图亮点；\n"
     "4. 【光线事故甄别】：严惩毫无正面补光控制、纯由展馆顶棚直射光在面部砸出死硬下沉阴影的恶性顶光。"),
    (9, "身形与穿帮专项", ["body"],
     "1. 身形比例评估（结合角色体型设定，评估机位俯仰带来的身形透视是否自然协调）。\n"
     "2. 严查服装穿帮：漫展入场手环未摘除、内衣肩带走光、安全裤露出、胸垫显形、拉链崩开。"),
]

DEFAULT_PASS2_PROMPT = """【当前任务】：执行 Pass 2 —— 全局合流带图交叉核验（Unified Cross-Check）。
你必须重新结合挂载的【全景图】与【1:1 脸脖切片】，站在宏观全局视角，对 Pass 1 产生的全部 9 步初检结果进行整体互查、因果消歧与误判推翻。

【物理遥测与现场背景】：{EXIF_TELEMETRY}
【Pass 1 全部9步初检结果汇总】：
{ALL_9_STEP_JSONS}

【全局交叉复核核心动作】：
1. 跨步因果消歧：结合全景图与脸脖切片，彻底理清跨步骤因果：
   - 斜构图复核：结合背景与前景倾斜度，交叉 Step 4（姿态动能），核查倾斜是否造成视觉失衡滑落感；
   - 脸脖色差消歧：结合 Step 8（主光角度与阴影），确认颈部暗黄是下巴自然投影还是底妆色差；
   - 磨皮消歧：结合漫展厚底妆常态与人脸像素占比，只要睫毛唇线边缘锐利，撤回无毛孔即磨皮过度的误判；
   - 动作与穿帮：神态、手势、服装细节与角色设定是否统一。
2. 推翻与修正：若某步判断有误或过于苛刻，在对应步骤的 "pass2_cross_check_notes" 中写明推翻/修正理由，并修正其 flaws/strengths。
3. 存疑升级：若某点按常规像失误但可能是摄影师刻意创作选择，放入对应步骤的 "ambiguous_for_user"。

【重要输出格式】：
严禁任何多余闲聊文字！必须严格输出一个 JSON 数组，包含全部 9 个步骤更新后的对象（每个对象的 status 必须更新为 2）。格式示意：
[
  {"step_id": 1, "step_name": "构图与场景", "status": 2, "applicable": true, "visual_evidence": [...], "strengths": [...], "flaws": [...], "ambiguous_for_user": [...], "pass2_cross_check_notes": "..."},
  ...
  {"step_id": 9, "step_name": "身形与穿帮专项", "status": 2, ...}
]"""

DEFAULT_PASS3_PROMPT = """【当前任务】：所有 9 步已完成 Pass 2 带图交叉核验（全员 status=2）。请结合挂载的全景图与下面 9 份终审 JSON，输出资深漫展摄影审片总监视角的最终报告。

【物理遥测与现场背景】：{EXIF_TELEMETRY}
【角色信息】：{CHARACTER_INFO}
【全部 9 步终审 JSON】：
{ALL_9_VERIFIED_JSONS}

请严格按以下 Markdown 结构输出（冷酷、客观、极客专业视角，拒绝和稀泥，亦拒绝盲目保守）：

### 🎯 场景定性与一句话结论
（先标明【拍摄场景：xxx（附依据）】，再从以下三项中**严格原样择一**输出定性（绝对严禁在此处混用夯/顶级/人上人等评级词！）：
- 【定性：直接进精选精修】（核心交付物完成度高、光影与神态扎实、无恶性穿帮即选此项，不要因背景常规而降级）
- 【定性：局部抢救可出片】（存在明显但后期可修复的重大瑕疵）
- 【定性：建议直接进废片篓】（存在不可逆技术或构图废片硬伤）
最后用一句话点破成败核心。）

### 📐 构图透视与动作体态深度解构（核心骨架必评）
（本章节为硬性解构项，必须从空间与力学两个维度深入剖析画面骨架）：
1. **【构图几何与空间透视解构】**：
   - **九宫格黄金分割与力点落位（三分法则）**：核心交付物（尤其是露出的主视眼、面部五官中心、关键持握道具）是否精准锚定在九宫格的黄金分割力点（三分线交点）上？分析头顶留白比例是否克制，视线朝向前方是否留有合理的呼吸与引导空间；
   - **重心构图与全画幅视觉天平（Visual Balance）**：深入分析全画面的视觉重心（Center of Gravity）落在何处。结合人物身形倾角、大面积暗部服饰的重量感、与特定高饱和色块分布，综合评判画面是底盘稳如泰山、受力均衡，还是重心跑偏导致画面向一侧失衡滑落或头重脚轻；
   - **几何构图骨架**：明确指出画面所采用的核心构图形式（如经典九宫格、三角稳定构图、对角线动势、中心冲击构图或视线引导线），评价构图形式对角色气场的加成效果；
   - **前中后景三层立体空间解构**：前景虚化构建纵深引导的质量、中景主体与背景的光学分离度、背景环境的光斑透视与世界观包覆感（严审画面是否具备通透呼吸感，还是扁平无层次）；
   - **画幅景别与机位俯仰**：精准定性景别（全身/七分身/半身/胸像/特写），尊重既定景别的完整边界（严禁对半身/胸像索要下肢或脚部）；结合平视/低仰/俯拍高度分析对人物比例的塑造；
   - **镜头自由度与边框截断**：读取 EXIF 镜头型号与当前焦段（区分定焦锁定、变焦最广端受限，还是变焦留有广角余量下的主动焦段取舍），再客观评判四周边缘对肢体的裁切合理性与留白呼吸感。
2. **【动作体态与力学张力解构】**：
   - **动作类型与叙事定性**：精准判定当前是站姿、坐姿、动姿还是特殊Pose，并说明其与角色设定的匹配度。严禁仅用“好看/唯美”等空洞词汇，必须指出动作的【物理支撑点】与【发力逻辑】，或者做了什么动作。
   - **重心平衡与受力线条**：分析站姿/坐姿/动姿的身体支撑点、脊柱曲度与双肩倾斜角度，动作是舒展从容、充满动态张力，还是失衡僵硬；
   - **体态微观力学**：大臂与躯干贴合度（是否压扁显壮）、耸肩缩脖、挤出赘肉或双下巴等体态力学瑕疵；
   - **动作指向与叙事说服力**：手部动作与手持武器/道具的指向性，动作是否与角色性格和现场氛围产生化学反应，定格瞬间是否具备强烈的感染力。

### 💡 光影架构与布光系统深度解构（立体光质必评）
（本章节为硬核光学解构项，必须从四维光路深度剖析现场布光与光质成色）：
1. **面光与眼神光塑造（Key Light & Catchlight）**：
   - 面部主光充盈度与曝光充分性，光质是柔和漫反射（带柔光箱/反光伞特征）还是硬光直射；
   - 瞳孔内是否存在清晰规整的眼神光（Catchlight），五官是否形成通透立体的明暗过渡而非惨淡大平光。
2. **空间分离度评估（轮廓光与景深虚化综合评判）**：
   - 必须从**“轮廓/发丝光勾勒”与“焦外光学虚化反差（锐利主体 vs 柔化背景）”**两个维度综合评价主体剥离度！严禁忽略镜头虚化作用而只盯着明度颜色算分离；
   - 评价轮廓光、发丝透光带来的空气感与边缘雕刻效果。
3. **现场环境光、冷暖分离与主题色温控制**：
   - 评估主光是否有效压制恶意杂光（如顶棚毒光）；
   - **尊重主题氛围与冷暖分离**：面光保持通透肤色还原、背景保留现场冷暖氛围色调，属于高阶色彩对比，结合画面主题客观评价其融合度，不得盲目要求强行统一面光与背景的色温。
4. **整体光比与影调骨架（Lighting Ratio & Chiaroscuro）**：
   - 全画面的主副光比控制，受光面与背光面的反差结构；
   - 暗部阴影是沉稳扎实、死而不焦且具有呼吸感过渡（Penumbra），还是发灰发平？明暗对照是否立得住画面的视觉冲击力。

### ✅ 客观亮点
（【动态裁决原则】：
- 若综合评级落入【💀 拉完了】：本节必须强制直接输出：`【无】毫无亮点，全身都是灾难，严禁强行美化找优点！`；
- 若综合评级落入【😐 NPC】：仅允许列出 1 条勉强及格的物理事实（如合焦未糊），没有就写无，严禁尬吹氛围；
- 若为【✨ 人上人】及以上档次：正常详尽列出光学控光、合焦锐度、色彩影调、神态气场与角色还原度亮点。二次元经典的斜刘海与侧颜发丝属于神态氛围加分项，绝不可算作缺点！）

### ⚠️ 客观瑕疵与救片/劝退裁决
（按【致命不可逆（不当机位俯仰导致严重比例不协调/半闭眼眨眼/手抖拉丝/恶性现实穿帮）】、【后期低成本可救（背景色块压暗/局部高光微调/微小穿帮消除）】、【微小瑕不掩瑜】分级列出。注意：长焦空间受限导致的肢体边缘切断属于物理妥协，列入瑕不掩瑜，不苛责摄影师）

### 🛠️ 后期与前期实战指南（按成片档次严格分流，拒绝无意义劳动）
【必须根据照片实际档次严格执行以下分支】：
- **若为【💀 拉完了】（放弃治疗·闲鱼警告）**：
  输出：`【放弃治疗·闲鱼警告】：你是不是拿快门故障、连拍废帧或手滑废片来找事情的？本图完全没有任何后期抢救的物理价值！拿去修纯属浪费电费。建议直接按 Shift+Delete 物理粉碎，顺便建议把相机镜头挂闲鱼出了，退坑保平安！`
- **若为【😐 NPC】（劝退警告·严禁浪费生命修图）**：
  输出：`【劝退警告·严禁浪费生命修图】：本图属于纯正的漫展路人随手抓拍级 NPC 废片。后期哪怕花两个小时抠图修形，也改变不了死亡机位和抓拍翻车的基本盘，投入产出比极低！请直接进废片篓归档，严禁浪费哪怕 1 分钟在上面！` 紧接着输出：`【📸 必须吸取的前期拍摄避坑指南（下次漫展怎么做人）】`，具体指出怎么根据角色身形调整协调的机位高度、现场怎么避开排灯毒光、怎么跟 Coser 倒数 321 避开眨眼、怎么避开杂乱现实穿帮。
- **若为【✨ 人上人】及以上（具备交付价值的正片）**：
  正常给出详实的【后期精准抢救思路】以及【下次拍摄进阶前期建议】。
  **【后期建议三大禁忌红线（严禁触犯）】：① 严禁建议强行提亮/拉平主光与轮廓光自然形成的下颌及颈部阴影；② 当主体已有虚化或轮廓分离时，严禁建议大幅压暗背景制造“抠图贴纸感”；③ 严禁脱离现场主题氛围，建议强行把面光拉得跟背景一样暖、或把背景漂得跟面光一样白！**

### 🧠 审美与风格主观锐评（结合器材性能与资历动态评判）
（针对烂片放开手脚犀利开喷，针对好片客观击节赞赏。深度结合本次机身、镜头参数、摄影师手持功底与现场可控条件，点评情绪张力与成片潜力。言之有物，不强行玩梗，亦不假正经。）

### 📊 综合评级与多维总结（结合器材看）
【评级标尺（以真实漫展实战为参照基准，拒绝拿商业无尘影棚的苛刻标准强压漫展场照；权重分明，不放水亦不压分）：🔥 夯 > 💎 顶级 > ✨ 人上人 > 😐 NPC > 💀 拉完了】
- 🔥 夯：大摄/漫展天花板级出片神作！在漫展实战中达到极高完成度：面部合焦刀锋级锐利、自主/借光光影极具立体塑形感（面光通透+轮廓剥离+光比高级）、神态气场具备强大穿透力与故事氛围，原图或微调即可作为代表作惊艳众人（允许存在不影响整体气场的微小边角瑕疵）；
- 💎 顶级：高级漫展摄影！明显与普通漫展场照拉开代差：具备清晰的主动控光意识（面光饱满、眼神光明亮、阴影有骨架）或极敏锐的借光构图嗅觉，参数决策冷静，角色神韵还原与动作定格扎实，整体质感拔群、瑕不掩瑜；
- ✨ 人上人：普通漫展合格摄影！能稳稳拿到“及格返图”的水准：对焦基本清楚、曝光没有重大失误、把人完整拍了下来，但光影往往偏平或被动依赖环境光（缺少主动光比塑形与高级眼神光）、构图与动作偏常规平铺、缺乏让人眼前一亮的视觉张力；
- 😐 NPC：路人随手按的游客照水准（如机位俯仰严重失调导致画面比例别扭、半闭眼眨眼、杂乱公共设施恶性穿帮、毫无光影与取舍意识）；
- 💀 拉完了：严重跑焦脱焦、快门手抖糊成残影、物理遮挡导致主体不可视的彻底废片。

【专业审片权重与主次分明原则（抓大放小）】：
1. **核心权重优先（大节不亏）**：以“面部合焦锐利度、主光光质、眼神神态传达、主体与背景的整体协调”为核心判定主干。漫展现场工况复杂，对于边缘微弱的受光不均、微小碎发或背景不可抗力色块等次要细节，应作为建设性的后期精进建议指出，不宜以偏概全、过度放大微瑕而盲目连降整体档次；
2. **客观看待布光贡献**：自主控光（如多灯布光、压制高感噪点、塑造细腻面光与眼神光）是显著的技术与画质加分项，应充分肯定其对成片质感的提升；若现场环境光恰好提供了优质光影，则应将光质红利公正归于场景，摄影师重在机位与借光嗅觉，各司其职；
3. **拒绝“木桶短板算术平均”，以主体最终呈现为准**：场景简陋或杂乱可独立给低分，不仅不应连带贬低摄影师与模特，更**严禁将【摄影师 + Coser + 场景】做算术平均来计算【照片本身】的得分！** 若摄影师以高超控光压制了杂乱背景、并与优秀模特共同呈现出极具冲击力的画面质感，成片评级必须向“最终视觉震撼度与主体交付高度”看齐，绝不能出现“顶级摄影拍顶级老师，成片却算普通人上人”的荒谬悖论；
4. **评论文意与评级绝对一致性**：严禁出现正文评价为“精修头牌、完美剥离、极高交付价值”，而最后给出的评级标签却落在“普通返图（人上人）”的自相矛盾分裂现象。

**【冲顶举证原则（严格区分“平庸及格”与“顶级冲夯”）】：**
   - **平庸归于【✨ 人上人】**：没有致命硬伤，但光影平淡、构图常规、缺乏突出视觉亮点与情绪张力、且有一定明显瑕疵的“平庸合格片”，也就是普通摄影的一般水准，无法和普通漫展场照拉开代差，本身就应停留在【✨ 人上人】，绝不可仅因“没犯错”就盲目拔高；
   - **【💎 顶级】冲刺【🔥 夯】必举证**：只有当某项维度（照片本身/摄影师/Coser老师）**已经具备突出的高水准亮点、稳稳达到【💎 顶级】及以上梯队时**，若未给到最高档【🔥 夯】，必须在紧随其后的评语中给出明确、具象、有分量的差距理由（如光影震撼度、氛围感染力或核心细节仍差临门一脚）；**若已达顶级水准且实在找不出实质性差距，严禁习惯性压分，必须痛快给到【🔥 夯】！**

1. **结合器材看**：深度结合 EXIF 硬件参数与摄影圈知名机身梗放开手脚自由发挥点评！
2. **照片本身**：【评级：🔥 夯 / 💎 顶级 / ✨ 人上人 / 😐 NPC / 💀 拉完了】+ 画面光影质感、视觉冲击力与成片完整度总结（作为独立成片，核心视听交付与光影质感为主导权重，不受常规漫展通道背景的算术拉低）。
3. **摄影师技术**：【评级：🔥 夯 / 💎 顶级 / ✨ 人上人 / 😐 NPC / 💀 拉完了】+ 曝光决策、机位嗅觉与现场控光能力总结。
4. **Coser 老师**：【评级：🔥 夯 / 💎 顶级 / ✨ 人上人 / 😐 NPC / 💀 拉完了】+ 角色神韵契合度、服化工艺·动作与镜头表现力总结。
5. **拍摄场景**：【评级：🔥 夯 / 💎 顶级 / ✨ 人上人 / 😐 NPC / 💀 拉完了】+ 现场背景纯净度、布景质感与光质底线总结。
   - 【场景客观定级准则】：
     - **【💀 拉完了】**：恶性穿帮占据主视觉（大面积路人围观、刺眼公共设施）、严重摩尔纹或全黑/死亡顶光彻底毁掉画面；
     - **【😐 NPC】**：背景明显杂乱（较多路人乱入、杂乱灯架/器材/设施穿帮较为刺眼）、或遭受明显顶光/杂光污染，对主体画面产生实质性干扰；
     - **【✨ 人上人】**：整体环境可用、无恶性死亡顶光毁图，允许存在**少量且对主体核心观感影响不大的边缘杂物或轻微干扰**；
     - **【💎 顶级】**：在复杂现场中做到**背景绝对纯净（几乎一点杂物、路人、穿帮都没有或不明显）**且无恶性光线破坏，或具备干净协调的官方主题布景与优质现场光赋能；
     - **【🔥 夯】**：具备专业摄影棚级布光质感、宏大精致的主题特装置景或神级舞台光效，场景本身即是震撼的视觉奇观。+ 现场布景质感与环境光质上限客观总结。

### ⚡ 省流总结（TL;DR）
（用最极客、最一针见血的大白话 2 句收尾：这片该不该要、最大的功臣/战犯是谁、值不值得修出正片。）

### ❓ 需要你亲自确认的存疑点（避免误判）
（【深度反问机制·严禁偷懒】：
- 仅当成片彻底落入【💀 拉完了】无法挽救时，才允许输出“事实清晰，无存疑点，建议直接删片”；
- **若为其他所有档次，严禁敷衍或写“无存疑点”！必须站在资深技术总监立场，向摄影师主动抛出 3~5 个具有专业探讨与决策分水岭价值的硬核问题（编号 1. 2. 3. 4. ...）**：
  ① 【空间与构图取舍】：例如前景虚化遮挡的入幅比例是否符合预期？特定边缘的裁切是否为既定景别设计？（严禁对明确的半身/胸像作品反问“有没有全身原图/为什么没拍到脚”！）
  ② 【影调与色彩倾向】：例如后期期望保留现场特定的环境氛围色调，还是校准为纯净的中性白基准？背景明度是压暗凸显主体还是保留通透？
  ③ 【动态与细节交付】：例如手持道具与动作姿态是否有特定叙事侧重？特写区域的景深与锐化强度倾向？
  ④ 【后期修图边界】：例如某些微弱环境色晕或特定造型细节，是希望作为现场真实光影保留，还是在后期中做纯净化处理？
提问必须直击像素事实与技术分支，言之有物，严禁泛泛而谈。）

【重要附带输出】：整篇 Markdown 报告全部写完后，另起一行严格输出分隔符 `===QUESTIONS_JSON===`，紧接着输出一个 JSON 字符串数组，内容与你上面 `### ❓` 章节列出的问题 1:1 完全一致（格式如：["1. ...", "2. ...", "3. ..."]，无问题则输出 `[]`）。
===QUESTIONS_JSON===
["1. ...", "2. ...", "3. ..."]"""

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
        # PySide6 原生色彩管理引擎：自动识别嵌入的 ICC (Display P3 / AdobeRGB 等) 并映射回标准 sRGB
        reader = QImageReader(path)
        reader.setAutoTransform(True)
        qimg = reader.read()
        if not qimg.isNull():
            # 若包含广色域 ICC 配置，内存级精准映射转换到 sRGB
            target_srgb = QColorSpace(QColorSpace.NamedColorSpace.SRgb)
            if qimg.colorSpace().isValid() and qimg.colorSpace() != target_srgb:
                qimg.convertToColorSpace(target_srgb)

            qimg = qimg.convertToFormat(QImage.Format_RGB888)
            w, h = qimg.width(), qimg.height()
            bpl = qimg.bytesPerLine()
            buf = qimg.bits()
            arr = np.frombuffer(buf, dtype=np.uint8, count=bpl * h).reshape(h, bpl)[:, : w * 3].reshape(h, w, 3)
            return cv2.cvtColor(np.array(arr), cv2.COLOR_RGB2BGR)
    except Exception:
        pass

    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
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

    # 缩放全景图至 1280px，保留清晰的背景文字与弱对比边缘，彻底消灭马赛克与色块误判
    scale_full = min(1.0, 1280.0 / max(orig_w, orig_h))
    bgr_full = cv2.resize(bgr, (int(orig_w * scale_full), int(orig_h * scale_full)), interpolation=cv2.INTER_AREA) if scale_full < 1.0 else bgr

    for path_out, mat in ((p_full, bgr_full), (p_fn, crop_face_neck), (p_hl, crop_hairline), (p_bd, crop_body)):
        ok, buf = cv2.imencode(".jpg", mat, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if ok:
            path_out.write_bytes(buf.tobytes())

    # 1280px 全景图 + 92 质量：既保证背景文字可辨认，又严控 ViT 切片数量，高画质低消耗
    b64_map = {
        "full": _encode_bgr_b64(bgr_full, max_dim=1280, quality=92),
        "face_neck": _encode_bgr_b64(crop_face_neck, max_dim=640, quality=90),
        "hairline": _encode_bgr_b64(crop_hairline, max_dim=640, quality=90),
        "body": _encode_bgr_b64(crop_body, max_dim=768, quality=88),
    }

    if ref_path and os.path.exists(ref_path):
        ref_bgr = _imread_bgr(ref_path)
        if ref_bgr is not None:
            b64_map["ref"] = _encode_bgr_b64(ref_bgr, max_dim=768, quality=85)

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

def _deepseek_search_intel(api_key: str, character_name: str, coser_cn: str, base_url: str = "") -> dict:
    """利用 DeepSeek /v1/tools/search 接口，联合检索角色人设与 Coser 官方出镜背景"""
    if not api_key:
        return {}

    search_url = "https://api.deepseek.com/v1/tools/search"
    if base_url and "deepseek.com" in base_url:
        search_url = base_url.rstrip("/").replace("/chat/completions", "").replace("/v1", "") + "/v1/tools/search"

    intel = {"character": "", "coser": ""}

    def _do_search(query_text: str) -> str:
        if not query_text or query_text.strip() in ("未指定", "未填写", "未知", "未指定（请结合画面自行识别或标明未知）"):
            return ""
        payload = {"query": query_text.strip()[:128], "count": 5, "need_content": False}
        req = urllib.request.Request(
            search_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            },
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                results = data.get("results", [])
                lines = []
                for r in results:
                    title = r.get("title", "")
                    snippet = r.get("summary") or r.get("snippet", "")
                    if snippet:
                        lines.append(f"- 【{title}】: {snippet[:200]}")
                return "\n".join(lines)
        except Exception as e:
            print(f"[DeepSeek Search Error]: {e}")
            return f"(检索请求异常: {e})"

    # 1. 角色人设检索
    if character_name:
        intel["character"] = _do_search(f"{character_name} 角色设定 外观 服饰 官方人设")

    # 2. 核心修正：绑定【老师名 + 角色名 + 官方 coser】，一枪打中官C身份！
    if coser_cn and coser_cn not in ("未填写", "未填写/未知", "未知"):
        query = f"{coser_cn} {character_name or ''} 官方 coser 漫展".strip()
        intel["coser"] = _do_search(query)

    return intel


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

            # 🌐 前置联网检索：利用免费 DeepSeek Search 查准角色人设与 Coser 背景，杜绝二次元认知偏差
            char_query = self.cfg.get("character_name", "").strip()
            coser_query = self.cfg.get("shooting_context", {}).get("coser_cn", "").strip()
            if char_query or (coser_query and coser_query not in ("未填写", "未填写/未知")):
                self.sig_log.emit(f"🌐 [联网先验] 正在调用 DeepSeek Search 检索角色【{char_query or '未知'}】与老师【{coser_query or '未知'}】情报...", "INFO")
                intel = _deepseek_search_intel(self.cfg["api_key"], char_query, coser_query, self.cfg.get("base_url", ""))
                if intel.get("character"):
                    telemetry["character_search_ground_truth"] = intel["character"]
                    self.sig_log.emit("✅ 官方角色人设已命中并锁定（发色/瞳色/特征先验就绪）", "SUCCESS")
                if intel.get("coser"):
                    telemetry["coser_search_ground_truth"] = intel["coser"]
                    self.sig_log.emit("✅ Coser 老师背景检索就绪（官方/知名身份已锁定）", "SUCCESS")
                # 重新刷盘最新的 meta.json
                (work_dir / "meta.json").write_text(json.dumps(telemetry, indent=2, ensure_ascii=False), encoding="utf-8")

            telemetry_str = json.dumps(telemetry, ensure_ascii=False)
            # 将搜索情报提拔为一等公民，组装成强约束的权威真理看板
            char_input_raw = telemetry.get("character_input", "未指定")
            char_intel = telemetry.get("character_search_ground_truth", "")
            coser_intel = telemetry.get("coser_search_ground_truth", "")

            char_blocks = [f"【目标角色与作品】：{char_input_raw}"]
            if char_intel:
                char_blocks.append(f"【DeepSeek 联网检索·官方人设权威基准（必须逐条比对）】：\n{char_intel}")
            if coser_intel:
                char_blocks.append(f"【DeepSeek 联网检索·Coser 老师背景情报】：\n{coser_intel}")
            if not char_intel and not coser_intel:
                char_blocks.append("（未检索到特定网络背景，请基于二次元经典人设通则客观审视）")

            char_str = "\n\n".join(char_blocks)
            sc = telemetry.get("shooting_context", {})
            ctrl_m_str = "可以现场沟通指导动作与神态" if sc.get("model_controllable", True) else "不可控（蹭拍/围观抓拍，无权指挥模特）"
            ctrl_l_str = "是（摄影师自己带灯自主打光/控光）" if sc.get("light_controllable", True) else "否（纯现场环境光/蹭展台灯，灯光不可控）"
            telemetry_str = (
                f"【现场先验与可控权限声明（必须严格采信）】：\n"
                f"- 展会与场景：{sc.get('expo_name', '未填写')} | 位置：{sc.get('scene_location', '自动识别')} | 老师CN：{sc.get('coser_cn', '未知')}\n"
                f"- 摄影师资历：{sc.get('photographer_level', '进阶')} | 模特可控权：{ctrl_m_str}\n"
                f"- 现场主光来源：{ctrl_l_str}\n\n"
                f"【底层物理与EXIF遥测JSON】：\n{telemetry_str}"
            )
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
                    f"【角色设定与联网检索基准】：\n{char_str}\n\n"
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

            # ---------------- Pass 2: 合流全局带图交叉核验 (1 次请求替代 9 次，Token 狂降 85%) ----------------
            self.sig_log.emit("🔍 [Pass 2] 门禁通过！启动合流带图全局交叉核验（1次高密推理替代9次重复传输）...", "INFO")
            all_p1_str = json.dumps([pass1_results[s[0]] for s in steps_cfg], indent=2, ensure_ascii=False)
            pass2_tpl = self.cfg["pass2_prompt"]
            user_txt = (
                pass2_tpl.replace("{EXIF_TELEMETRY}", telemetry_str)
                         .replace("{ALL_9_STEP_JSONS}", all_p1_str)
            )

            # Pass 2 仅挂载 960px 全景图 + 1:1 脸脖切片（若有参考立绘则附带），兼具宏观与微观，杜绝 27 次重复图片搬运
            keys_p2 = [k for k in ("full", "face_neck", "ref") if k in b64_map]
            imgs_p2 = [(img_name_cn.get(k, k), b64_map[k]) for k in keys_p2]

            raw_p2 = _call_vlm_api(base_url, api_key, model, sys_prompt, user_txt, imgs_p2)
            if isinstance(raw_p2, tuple):
                raw_p2 = raw_p2[0]

            # 解析合流 JSON 数组（兼容直接输出数组或包裹在 {"steps": [...]} 内）
            pass2_items_map = {}
            try:
                s_clean = raw_p2.strip()
                if s_clean.startswith("```"):
                    lines = s_clean.splitlines()
                    if lines and lines[0].startswith("```"): lines = lines[1:]
                    if lines and lines[-1].strip().startswith("```"): lines = lines[:-1]
                    s_clean = "\n".join(lines).strip()

                parsed = json.loads(s_clean)
                if isinstance(parsed, dict):
                    for k in ("steps", "results", "data", "pass2_results"):
                        if isinstance(parsed.get(k), list):
                            parsed = parsed[k]
                            break
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict) and "step_id" in item:
                            pass2_items_map[int(item["step_id"])] = item
            except Exception:
                # 极端容错：正则扫出每个独立 JSON 对象
                decoder = json.JSONDecoder()
                pos = 0
                while True:
                    idx = raw_p2.find("{", pos)
                    if idx < 0: break
                    try:
                        obj_p, end_idx = decoder.raw_decode(raw_p2, idx)
                        if isinstance(obj_p, dict) and "step_id" in obj_p:
                            pass2_items_map[int(obj_p["step_id"])] = obj_p
                        pos = end_idx
                    except Exception:
                        pos = idx + 1

            pass2_results = dict(pass1_results)
            for sid, sname, _, _ in steps_cfg:
                obj_final = pass2_items_map.get(sid) or pass1_results[sid]
                obj_final["step_id"] = sid
                obj_final["step_name"] = sname
                obj_final["status"] = 2
                pass2_results[sid] = obj_final

                # 完美保留原来的文件落盘与 UI 跳灯，外部感知完全一致！
                (work_dir / f"step_{sid}_pass2.json").write_text(json.dumps(obj_final, indent=2, ensure_ascii=False), encoding="utf-8")
                self.sig_step_updated.emit(sid, 2, obj_final)

            self.sig_log.emit("🛡️ [Pass 2] 全局合流交叉核验完成！9 步状态全部推进至 status=2", "SUCCESS")

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

            # 提前在括号外提取好联网检索基准字符串，杜绝语法解析冲突
            char_intel = self.telemetry.get("character_search_ground_truth", "")
            coser_intel = self.telemetry.get("coser_search_ground_truth", "")
            search_ground_txt = ""
            if char_intel or coser_intel:
                search_ground_txt = "【官方人设与 Coser 联网检索权威基准（不可推翻的先验事实）】：\n"
                if char_intel:
                    search_ground_txt += f"官方角色人设：\n{char_intel}\n"
                if coser_intel:
                    search_ground_txt += f"Coser 老师背景：\n{coser_intel}\n"
                search_ground_txt += "\n"

            user_txt = (
                f"【上一轮审片报告】：\n{self.report_md}\n\n"
                f"{search_ground_txt}"
                f"【原始器材与物理遥测数据】：\n{telemetry_str}\n\n"
                f"【现场可控权限声明】：\n"
                f"- 老师/模特能否现场沟通摆拍：{'可以沟通指导' if ctrl_model else '不可控（蹭拍/路人抓拍/无权限指挥）'}\n"
                f"- 现场主光是否为摄影师自己打的：{'是（自己布光）' if ctrl_light else '否（现场环境光/蹭别人灯，灯光不可控）'}\n\n"
                f"【摄影师问卷逐条反馈】：\n{qa_block}\n\n"
                f"【摄影师指出 AI 理解错的地方 / 额外说明】：\n{misread_note or '无'}\n\n"
                f"【终审校准铁律（独立技术审计与有效举证原则）】：\n"
                f"1. **【视觉像素事实绝对优先（Pixel Ground Truth）】**：\n"
                f"   - 输入图像的肉眼可见像素是最高裁决依据，用户的文字反馈仅作为现场物理约束的补充参考，绝不具备直接推翻视觉事实的效力；\n"
                f"   - 严禁出现 LLM 常见的“为维持对话和谐而顺从用户、盲目道歉并妥协结论”的谄媚倾向（Anti-Sycophancy）。审片总监必须恪守专业标准，独立判定。\n"
                f"2. **【有效举证与无效抗辩分流（核心裁决准则）】**：\n"
                f"   - **有效举证（准予修正/推翻原判）**：\n"
                f"     ① 【客观环境与器材死局】：因定焦焦段无法变焦、物理围栏阻隔退无可退、或官方展台/联动品牌既定布景资产限制的，采纳现场物理限制并修正指责成因（例如：严禁建议压暗官方品牌涂鸦或改变官方原设配色）；\n"
                f"     ② 【前景构图与叙事道具（严格区分杂物）】：若某视觉元素此前被初检误判为‘杂物干扰’，但摄影师确认其属于【刻意布置的前景虚化、角色自身服饰/道具的纵深延伸、或与背景呼应的色块节奏】：只要该元素没有粗暴切碎主体核心面部五官，且在视觉上确实起到了增加画幅纵深与氛围感的作用，必须定性为成立的构图设计，彻底撤回‘杂物遮挡/穿帮’指控！\n"
                f"     ③ 【物理光影常识】：暗部（黑衣、肢体遮挡背光面）该暗就得暗，光影透射形成的色温层次属于真实光学物理，不得无理要求磨平或提亮破坏骨架。\n"
                f"   - **无效抗辩（坚决驳回并维持原判）**：\n"
                f"     凡是将【硬性技术翻车】（如：眼睑下垂视线溃散的抓拍废帧、严重手抖糊片、未做补光的死亡顶光阴影）以及【与二次元叙事彻底割裂的恶性现实穿帮】（如：公共设施警示牌、未摘除的展会入场纸手环、乱入路人、现场安防器材）强行归因为“个人主观风格”或辩称“我觉得好看”的，一律判定为无效抗辩！现实低级穿帮与生理性技术失误绝不容许以艺术之名洗白。\n"
                f"3. **【现场可控性与建议适配】**：\n"
                f"   - 严格遵循摄影师勾选的现场权限声明：若已声明模特/灯光不可控，建议中严禁出现要求现场重摆或调整灯位的无效指导；若已声明为特定景别（如半身大头）且动作逻辑自洽（如抱臂遮挡手部），不得强行指责未露出的部位。\n"
                f"4. {speech_rule}\n\n"
                f"【强制输出结构（严格按此 Markdown 章节顺序输出）】：\n"
                f"### 🎯 终审校准定性\n"
                f"### 🔍 问卷反馈逐条裁决\n"
                f"### ✅ 修正后的亮点\n（结合用户确认的真实设定与现场条件，重新梳理真正成立的亮点）\n"
                f"### ⚠️ 修正后的缺点\n（剔除已证实是 AI 误判的项，保留所有客观成立及“尊重但保留意见”的缺点，分级列出）\n"
                f"### 🛠️ 后期抢救与重构指南\n"
                f"{'### 🗣️ 现场跟老师沟通的话术' + chr(10) if ctrl_model else ''}"
                f"### 📊 照片的总体评价（结合器材看）\n"
                f"【综合评级标尺（五档严格定级，禁止和稀泥）：🔥 夯 > 💎 顶级 > ✨ 人上人 > 😐 NPC > 💀 拉完了】\n\n"
                f"1. **结合器材看**：深度结合 EXIF 硬件参数，大胆调动【摄影圈知名刻板印象、机身梗、年份身价与色彩玄学】现场自由发挥点评！\n"
                f"   - 允许调动圈内熟知的器材人设（如 R50 塑料入门机被戏谑狗都不买、索尼黄与对焦怪物、佳能红润感动常在、富士溢价理财、老单反搬砖传家宝、适马健身器材等）；\n"
                f"   - 必须把器材的“圈内名声/价格身价”与本次画面里的“实际成色”挂钩：是穷鬼战神绝地逆袭，还是百万富翁盲人摸象？若低配超常发挥，尽情赞叹其把玩具榨成生产力的技术定力；若顶配翻车，允许用符合摄影师性格的犀利吐槽开喷（如建议挂闲鱼）再注明“开个玩笑，说正经的：”切回专业复盘。言之有物，严禁干瘪报菜名。\n"
                f"2. **照片本身总结**：【评级：🔥 夯 / 💎 顶级 / ✨ 人上人 / 😐 NPC / 💀 拉完了】+ 独立成片完整度、视觉张力与交付价值。\n"
                f"3. **摄影师技术总结**：【评级：🔥 夯 / 💎 顶级 / ✨ 人上人 / 😐 NPC / 💀 拉完了】+ 结合可控/不可控，评价曝光、机位、抓拍时机与后期审美取舍。\n"
                f"4. **Coser 老师总结**：【评级：🔥 夯 / 💎 顶级 / ✨ 人上人 / 😐 NPC / 💀 拉完了】+ 评价老师角色还原、服化细节、镜头表现与肢体情绪。\n"
                f"5. **场景总结**：【评级：🔥 夯 / 💎 顶级 / ✨ 人上人 / 😐 NPC / 💀 拉完了】+ 评价场地布景、光质杂乱度及对出片上限的物理限制。\n\n"
                f"### ⚡ 省流总结（TL;DR）\n"
                f"（用最极客、最一针见血的大白话 2~3 句收尾：这片能不能要、摄影师和老师到底谁带飞了谁/谁拉了跨、后期花几分钟值不值得修。）"
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
        if full_p and os.path.exists(full_p):
            mat = _imread_bgr(full_p)
            if mat is not None:
                self.last_b64_full = _encode_bgr_b64(mat, max_dim=1280, quality=92)
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