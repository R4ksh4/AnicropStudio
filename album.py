from sony_af import extract_sony_af_points, _sony_af_core, _sony_af_from_jpeg_bytes
from nikon_af import _nikon_af_core
import os
import sys
import math
import struct
import hashlib
import re
from collections import OrderedDict
import time
import shutil
from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QFileDialog, QListView, QAbstractItemView, QStyledItemDelegate, 
    QStyle, QFrame, QStackedWidget, QProgressBar, QSlider, 
    QLineEdit, QMenu, QCheckBox, QMessageBox, QDialog, QApplication, QComboBox,
    QGridLayout
)
from PySide6.QtCore import (
    Qt, QSize, QRect, QRectF, QSizeF, QPoint, QPointF, QThreadPool, QRunnable, 
    QAbstractListModel, QModelIndex, Signal, Slot, QObject, QThread, QSettings, QEvent, QTimer
)
from focus_inspector import FocusInspector
from rating_writer import write_photo_rating
from sony_af import extract_sony_af_points, _sony_af_core, _sony_af_from_jpeg_bytes
from PySide6.QtGui import (
    QPainter, QColor, QFont, QPen, QBrush, QPixmap, 
    QImage, QImageReader, QFontMetrics, QKeyEvent, QMouseEvent, 
    QWheelEvent, QCursor, QTransform, QAction, QPainterPath
)

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ══════════════════════════════════════════════════════════════
# 【快捷键中央注册表】动作 -> (默认键, 功能描述, 固定备用键)
# 用户可在 设置 -> 快捷键 中改绑 A~Z 字母键；QSettings 持久化，即时生效
# ══════════════════════════════════════════════════════════════
DEFAULT_SHORTCUTS = {
    "prev":           ("A", "单图 - 上一张照片", "←"),
    "next":           ("D", "单图 - 下一张照片", "→"),
    "cmp_l_prev":     ("U", "对比 - 左图上一张", ""),
    "cmp_l_next":     ("I", "对比 - 左图下一张", ""),
    "cmp_r_prev":     ("O", "对比 - 右图上一张", ""),
    "cmp_r_next":     ("P", "对比 - 右图下一张", ""),
    "mark":           ("H", "标记 / 取消标记 (选片/废片)", "X / Delete / S"),
    "eye_zoom":       ("R", "眼部特写 开/关", "鼠标双击"),
    "af":             ("F", "对焦点显示 开/关", ""),
    "compare":        ("C", "对比模式 开/关", ""),
    "grid":           ("Q", "构图辅助线 循环切换", "Shift+Q 翻转螺旋象限"),
    "precrop":        ("W", "预裁剪遮罩 循环切换", ""),
}


# 系统保留快捷键（与体系深度绑定，仅展示不可修改）
FIXED_SHORTCUTS = [
    ("Esc", "退出沉浸全屏 / 返回画廊"),
    ("Space", "沉浸全屏 开/关"),
    ("↑ ↓ ← →", "画面上下左右平移漫游"),
    ("0 ~ 5", "打星：0=清除，1~5=星级（与 DPP / Lightroom 互通）"),
    ("Ctrl + 滚轮", "以光标为锚点顺滑缩放"),
    ("滚轮 / 左键拖拽", "平移画面漫游"),
]

RAW_IMAGE_EXTS = {".cr3", ".cr2", ".arw", ".nef", ".dng", ".orf", ".rw2"}

def is_raw_file(file_path: str) -> bool:
    return Path(file_path).suffix.lower() in RAW_IMAGE_EXTS

def _extract_cr2_embedded_jpeg(data: bytes) -> bytes:
    """
    佳能 CR2 (Canon RAW v2) 全尺寸满血原图定点提取引擎
    佳能 5D3 / 5D4 / 6D 核心机理:
      IFD0 (固定偏移 16) 下的 Tag 0x0111 (StripOffsets) 和 Tag 0x0117 (StripByteCounts)
      存储的正是 5760x3840 的全尺寸满血大图 (PreviewImage，体积通常 1MB~3.5MB)！
    """
    if len(data) < 64 or data[:4] != b'II*\x00' or data[8:12] != b'CR\x02\x00':
        return b""

    endian = '<'
    try:
        ifd0_off = struct.unpack(endian + 'I', data[4:8])[0]
        if ifd0_off <= 0 or ifd0_off + 2 > len(data):
            return b""

        num_entries = struct.unpack(endian + 'H', data[ifd0_off : ifd0_off + 2])[0]
        if num_entries <= 0 or ifd0_off + 2 + num_entries * 12 > len(data):
            return b""

        t_off, t_len = 0, 0
        for i in range(min(num_entries, 60)):
            ep = ifd0_off + 2 + i * 12
            tag = struct.unpack(endian + 'H', data[ep : ep + 2])[0]
            val = struct.unpack(endian + 'I', data[ep + 8 : ep + 12])[0]
            if tag == 0x0111:      # StripOffsets (5760x3840 满血大图起始偏移)
                t_off = val
            elif tag == 0x0117:    # StripByteCounts (大图字节长度)
                t_len = val

        if t_off > 0 and t_len > 1024 and t_off + t_len <= len(data):
            cand = data[t_off : t_off + t_len]
            if cand.startswith(b'\xff\xd8'):
                return cand
    except Exception:
        pass

    return b""


def _scan_embedded_jpeg(data: bytes) -> bytes:
    """【不解析 RAW】直接在二进制流中扫描内嵌 JPEG。对 CR2 走定点解剖，其他走最长段扫描。"""
    # 优先拦截佳能单反 CR2，毫秒级提取 IFD1 标准预览
    cr2_preview = _extract_cr2_embedded_jpeg(data)
    if cr2_preview:
        return cr2_preview

    best = b""
    pos = 0
    n = len(data)
    while pos < n - 4:
        soi = data.find(b'\xff\xd8\xff', pos)
        if soi < 0:
            break
        eoi = data.find(b'\xff\xd9', soi + 3)
        if eoi < 0:
            break
        seg = data[soi : eoi + 2]
        if len(seg) > len(best):
            best = seg
        pos = eoi + 2
    return best

def _default_exif_res() -> dict:
    return {"camera": "未知机身", "lens": "未知镜头", "focal": "--", "aperture": "--",
            "shutter": "--", "iso": "--", "ev": "--", "date": "--",
            "size": "--", "dimensions": "--"}

# ---------- RAW 单趟读取缓存 ----------
# 同一 RAW 文件只从盘上读一趟：内嵌预览 / 方向 / EXIF / AF 框 / XMP 星级 全部从这一趟派生，
# LRU 缓存（6 份，只存派生结果，原始大缓冲用完即弃）；mtime 变化自动失效（打星写回后安全）。
# 此前看一张 RAW 要整读文件 4~5 趟（96+16+96+16MB），系统盘缓存命中后纯属空烧 CPU。
_RAW_BUNDLE_CACHE = OrderedDict()
_RAW_BUNDLE_MAX = 6

def _orientation_from_candidates(cands) -> int:
    """沿 TIFF 候选的 IFD0→IFD1 链找 0x0112 方向；读不到返回 1"""
    for tiff, endian in cands:
        try:
            ifd = struct.unpack(endian + 'I', tiff[4:8])[0]
            visited = 0
            while 0 < ifd and ifd + 2 <= len(tiff) and visited < 8:
                visited += 1
                num = struct.unpack(endian + 'H', tiff[ifd : ifd + 2])[0]
                if num <= 0 or num > 200 or ifd + 2 + num * 12 + 4 > len(tiff):
                    break
                for k in range(num):
                    ep = ifd + 2 + k * 12
                    if struct.unpack(endian + 'H', tiff[ep : ep + 2])[0] == 0x0112:
                        val = struct.unpack(endian + 'H', tiff[ep + 8 : ep + 10])[0]
                        if val in (1, 2, 3, 4, 6, 8):
                            return val
                ifd = struct.unpack(endian + 'I', tiff[ifd + 2 + num * 12 : ifd + 2 + num * 12 + 4])[0]
        except Exception:
            continue
    return 1

# ---- CR3 盒结构走读（轻量路径专用）：只按 ISOBMFF 盒头跳转，不整读文件 ----
# 结构依据 lclevy/canon_cr3：顶层 ftyp/moov/uuid(XMP)/uuid(PRVW 预览)/mdat；
# XMP 在 uuid=be7acfcb…，预览在 uuid=eaf42b5e… 内的 PRVW 盒（+20 jpeg_size，+24 jpeg 数据），
# moov 内 uuid=85c0b687… 装 CMT1..4（CMT1=TIFF IFD0，含 0x0112 方向）。
_CR3_UUID_XMP = bytes.fromhex('be7acfcb97a942e89c71999491e3afac')
_CR3_UUID_PREVIEW = bytes.fromhex('eaf42b5e1c984b88b9fbb7dc406e4d16')
_CR3_UUID_CANON = bytes.fromhex('85c0b687820f11e08111f4ce462b6a48')

def _cr3_read_boxes(f, start, end, fsize):
    """生成 (type, payload_off, payload_size, box_end)；只读 8/16 字节盒头"""
    pos = start
    while pos + 8 <= min(end, fsize):
        f.seek(pos)
        hdr = f.read(8)
        if len(hdr) < 8:
            break
        size, btype = struct.unpack('>I4s', hdr)
        hlen = 8
        if size == 1:
            ext = f.read(8)
            if len(ext) < 8:
                break
            size = struct.unpack('>Q', ext)[0]
            hlen = 16
        elif size == 0:
            size = fsize - pos
        if size < hlen or pos + size > fsize + 16:
            break
        yield btype, pos + hlen, size - hlen, pos + size
        pos += size

def _cr3_light(file_path: str) -> dict:
    """CR3 轻量解剖：盒走读取出 PRVW 预览 JPEG + XMP 星级 + CMT1 方向。
    全程只读 moov/uuid 小区间（通常 <2MB），不碰 mdat。失败返回空 dict 由调用方回退整读。"""
    out = {"embedded": b"", "rating": 0, "orientation": 0}
    try:
        fsize = os.path.getsize(file_path)
        with open(file_path, 'rb') as f:
            for btype, off, blen, bend in _cr3_read_boxes(f, 0, fsize, fsize):
                if btype == b'moov' and not out["orientation"]:
                    # moov 内的佳能私有 uuid 装 CMT1..4（前面隔着 CNCV/CCTP/CTBO/free 若干盒，
                    # 探头定位不可靠）——直接按盒头枚举子盒找 CMT1（=TIFF IFD0，含 0x0112 方向）
                    for t2, o2, l2, e2 in _cr3_read_boxes(f, off, off + blen, fsize):
                        if t2 != b'uuid':
                            continue
                        cmt1 = b""
                        for t3, o3, l3, e3 in _cr3_read_boxes(f, o2 + 16, o2 + l2, fsize):
                            if t3 == b'CMT1':
                                f.seek(o3)
                                cmt1 = f.read(min(l3, 4 * 1024 * 1024))
                                break
                        if cmt1:
                            endian = '<' if cmt1[:2] == b'II' else ('>' if cmt1[:2] == b'MM' else None)
                            if endian:
                                out["orientation"] = _orientation_from_candidates([(cmt1, endian)])
                            break
                elif btype == b'uuid':
                    # 不依赖精确 16 字节 usertype（新机型可能变体）：
                    # 探头 4KB 按内容特征归类——含 'PRVW' 是预览盒，含 '<x:xmpmeta' 是 XMP 盒
                    f.seek(off)
                    probe = f.read(min(blen, 4096))
                    is_prvw = b'PRVW' in probe
                    is_xmp = (not is_prvw) and b'<x:xmpmeta' in probe
                    if is_prvw and not out["embedded"]:
                        # usertype(16B) 之后通常还有 8 字节前缀（00000000 00000001）才是 PRVW 盒；
                        # 不假定前缀长度，直接在探头内定位 'PRVW' 特征再回退 4 字节取盒头
                        k = probe.find(b'PRVW')
                        if k >= 4:
                            box_start = off + k - 4
                            bsize = struct.unpack('>I', probe[k - 4 : k])[0]
                            head = probe[k + 4 : k + 20]  # payload 内: +0 unknown, +12 jpeg_size, +16 jpeg 数据
                            if len(head) < 16:
                                f.seek(box_start + 8)
                                head = f.read(16)
                            if len(head) == 16:
                                jsize = struct.unpack('>I', head[12:16])[0]
                                if 1024 < jsize <= min(bsize - 24, 32 * 1024 * 1024):
                                    f.seek(box_start + 24)
                                    jpg = f.read(jsize)
                                    if jpg[:2] == b'\xff\xd8':
                                        out["embedded"] = jpg
                    elif is_xmp and not out["rating"]:
                        xmp = probe
                        if blen > 4096:
                            f.seek(off)
                            xmp = f.read(min(blen, 262144))
                        m = re.search(rb'xmp:Rating\s*=\s*"\s*(-?\d)\s*"', xmp) or \
                            re.search(rb'<xmp:Rating>\s*(-?\d)\s*</xmp:Rating>', xmp)
                        if m:
                            out["rating"] = max(0, min(5, int(m.group(1))))
                elif btype == b'mdat':
                    break  # mdat 是 RAW 主数据，轻量路径永不进
                if out["embedded"] and out["orientation"]:
                    break
    except Exception:
        return {}
    return out

def _cr3_cmt_blocks(file_path: str) -> list:
    """CR3 盒走读提取 CMT1..4 权威 EXIF 块（[(payload, endian), ...]）。
    CMT1=IFD0(机身/方向) CMT2=ExifIFD(快门/光圈/ISO/焦距/EV) CMT3=MakerNote。
    全文 TIFF 签名扫描会撞上假块/错基准块（R50 快门锁死 1/842s 的元凶），CR3 一律走这里。"""
    blocks = []
    try:
        fsize = os.path.getsize(file_path)
        with open(file_path, 'rb') as f:
            for btype, off, blen, bend in _cr3_read_boxes(f, 0, fsize, fsize):
                if btype == b'mdat':
                    break
                if btype != b'moov':
                    continue
                for t2, o2, l2, e2 in _cr3_read_boxes(f, off, off + blen, fsize):
                    if t2 != b'uuid':
                        continue
                    cmts = {}
                    for t3, o3, l3, e3 in _cr3_read_boxes(f, o2 + 16, o2 + l2, fsize):
                        if t3.startswith(b'CMT') and l3 > 16:
                            f.seek(o3)
                            payload = f.read(min(l3, 8 * 1024 * 1024))
                            endian = '<' if payload[:2] == b'II' else ('>' if payload[:2] == b'MM' else None)
                            if endian:
                                cmts[t3.decode(errors='ignore')] = (payload, endian)
                    for name in ('CMT1', 'CMT2', 'CMT3', 'CMT4'):
                        if name in cmts:
                            blocks.append(cmts[name])
                    if blocks:
                        return blocks
    except Exception:
        pass
    return blocks

def _raw_bundle(file_path: str, light: bool = False) -> dict:
    """RAW 单趟读取 + 派生（预览/方向/EXIF/AF/星级），LRU 缓存。
    light=True 只派生预览/方向/星级（供缩略图批量遍历），跳过 TIFF 候选枚举/EXIF/AF 等 CPU 密集派生；
    后续完整调用会在同一缓存条目上补齐。"""
    try:
        mt = os.path.getmtime(file_path)
    except Exception:
        return {}
    ent = _RAW_BUNDLE_CACHE.get(file_path)
    if ent is not None and ent.get("mtime") == mt:
        _RAW_BUNDLE_CACHE.move_to_end(file_path)
        if light or not ent.get("light"):
            return ent
        # 命中的是轻量条目但需要完整数据：落到下面重算补齐
    bundle = {"mtime": mt, "embedded": b"", "orientation": 1, "exif": None, "af": [], "rating": 0, "light": light}
    try:
        if light and file_path.lower().endswith('.cr3'):
            # CR3 轻量快车道：盒走读只取 PRVW 预览/XMP 星级/CMT1 方向（通常 <2MB），不整读文件
            lite = _cr3_light(file_path)
            if lite.get("embedded"):
                bundle["embedded"] = lite["embedded"]
                bundle["rating"] = lite.get("rating", 0)
                o = lite.get("orientation", 0) or _jpeg_exif_orientation(lite["embedded"])
                bundle["orientation"] = o or 1
                _RAW_BUNDLE_CACHE[file_path] = bundle
                _RAW_BUNDLE_CACHE.move_to_end(file_path)
                while len(_RAW_BUNDLE_CACHE) > _RAW_BUNDLE_MAX:
                    _RAW_BUNDLE_CACHE.popitem(last=False)
                return bundle
        with open(file_path, 'rb') as f:
            data = f.read(96 * 1024 * 1024)

        embedded = _scan_embedded_jpeg(data)
        bundle["embedded"] = embedded

        idx = data.find(b'<x:xmpmeta')
        if idx >= 0:
            m = re.search(rb'xmp:Rating\s*=\s*"\s*(-?\d)\s*"', data[idx: idx + 65536]) or \
                re.search(rb'<xmp:Rating>\s*(-?\d)\s*</xmp:Rating>', data[idx: idx + 65536])
            if m:
                bundle["rating"] = max(0, min(5, int(m.group(1))))

        # 1. 优先检查同名 .xmp 伴侣文件 (Lightroom / Bridge / 达芬奇互通)
        if bundle["rating"] == 0:
            p_raw = Path(file_path)
            for sidecar in (p_raw.with_suffix('.xmp'), p_raw.with_suffix('.XMP'), p_raw.parent / (p_raw.name + '.xmp')):
                if sidecar.exists():
                    try:
                        s_txt = sidecar.read_bytes()
                        m = re.search(rb'xmp:Rating\s*=\s*"\s*(-?\d)\s*"', s_txt) or \
                            re.search(rb'<xmp:Rating>\s*(-?\d)\s*</xmp:Rating>', s_txt)
                        if m:
                            bundle["rating"] = max(0, min(5, int(m.group(1))))
                            break
                    except Exception:
                        pass

        # 2. 索尼微单硬件级绝杀：解剖 Sony MakerNote Tag 0x2002 (机身回放打星)
        if bundle["rating"] == 0 and file_path.lower().endswith('.arw'):
            pos = 0
            while True:
                idx_sony = data.find(b'SONY DSC', pos, min(len(data), 4 * 1024 * 1024))
                if idx_sony == -1:
                    break
                if idx_sony + 14 <= len(data):
                    cnt = struct.unpack_from('<H', data, idx_sony + 12)[0]
                    if 0 < cnt <= 512:
                        for i in range(cnt):
                            ep = idx_sony + 14 + i * 12
                            if ep + 12 > len(data):
                                break
                            tag = struct.unpack_from('<H', data, ep)[0]
                            if tag == 0x2002:
                                r_val = struct.unpack_from('<I', data, ep + 8)[0]
                                if 1 <= r_val <= 5:
                                    bundle["rating"] = r_val
                                    break
                        if bundle["rating"] > 0:
                            break
                pos = idx_sony + 8

        cands = list(_iter_raw_tiff_candidates(data))
        is_tiff_raw = file_path.lower().endswith(('.arw', '.cr2', '.nef', '.dng'))
        if is_tiff_raw:
            # 原生 TIFF 架构 (ARW/CR2/NEF/DNG): 容器 IFD0 记录真实物理陀螺仪方向
            c_orient = _orientation_from_candidates(cands)
            bundle["orientation"] = c_orient if c_orient in (1, 2, 3, 4, 6, 8) else 1
        else:
            orient = _jpeg_exif_orientation(embedded) if embedded else 0
            bundle["orientation"] = orient if orient else _orientation_from_candidates(cands)

        if light:
            _RAW_BUNDLE_CACHE[file_path] = bundle
            _RAW_BUNDLE_CACHE.move_to_end(file_path)
            while len(_RAW_BUNDLE_CACHE) > _RAW_BUNDLE_MAX:
                _RAW_BUNDLE_CACHE.popitem(last=False)
            return bundle

        res = _default_exif_res()
        # CR3 优先走 CMT 权威块（顺序固定、偏移基准正确）；扫不到/字段缺再回落签名扫描补齐
        if file_path.lower().endswith('.cr3'):
            for payload, endian in _cr3_cmt_blocks(file_path):
                _fill_exif_from_tiff(res, payload, endian)
        if res["camera"] == "未知机身" or res["aperture"] == "--" or res["lens"] == "未知镜头":
            for tiff, endian in cands:
                _fill_exif_from_tiff(res, tiff, endian)
                if res["camera"] != "未知机身" and res["aperture"] != "--" and res["lens"] != "未知镜头":
                    break
        bundle["exif"] = res

        o = bundle["orientation"]
        # 【多品牌】对焦点提取：双通道兜底管线
        # 路径 1: 优先从内嵌预览 JPEG 尝试提取 (部分机型内嵌带 MakerNote)
        boxes = []
        if embedded:
            etiff = _exif_tiff_block(embedded)
            if etiff is not None:
                vendor = _makernote_vendor(etiff)
                if vendor == b'SONY':
                    boxes = _sony_af_core(etiff, '<' if etiff[:2] == b'II' else '>', o)
                elif vendor == b'NIKON':
                    boxes = _nikon_af_core(etiff, '<' if etiff[:2] == b'II' else '>', o)
                else:
                    boxes = _canon_af_core(etiff, '<' if etiff[:2] == b'II' else '>', o)

        # 路径 2: 内嵌预览无 MakerNote 时，直接解剖 RAW 容器自身 (索尼 ARW / 尼康 NEF / 佳能 CR3)
        if not boxes:
            for tiff, endian in cands:
                # 尝试索尼 ARW 容器本身
                boxes = _sony_af_core(tiff, endian, o)
                if boxes:
                    break
                # 尝试尼康 NEF 容器本身 (Tag 0x00B7 AFInfo2)
                boxes = _nikon_af_core(tiff, endian, o)
                if boxes:
                    break
                # 尝试佳能原生/独立块
                boxes = _canon_af_core(tiff, endian, o)
                if boxes:
                    break

        bundle["af"] = boxes or []
    except Exception:
        pass
    _RAW_BUNDLE_CACHE[file_path] = bundle
    _RAW_BUNDLE_CACHE.move_to_end(file_path)
    while len(_RAW_BUNDLE_CACHE) > _RAW_BUNDLE_MAX:
        _RAW_BUNDLE_CACHE.popitem(last=False)
    return bundle

def extract_raw_embedded_jpeg(file_path: str, max_scan: int = 96 * 1024 * 1024, light: bool = False) -> bytes:
    """【不解析 RAW】取机内全尺寸预览 JPEG（走单趟缓存；max_scan 仅为调用兼容保留）"""
    try:
        return _raw_bundle(file_path, light=light).get("embedded", b"")
    except Exception:
        return b""

def _jpeg_exif_orientation(data: bytes) -> int:
    """从一段 JPEG 字节流的 APP1/EXIF IFD0 里读 0x0112 方向；读不到返回 0"""
    try:
        if len(data) < 4 or data[:2] != b'\xFF\xD8':
            return 0
        idx = 2
        while idx < len(data) - 4:
            if data[idx] != 0xFF: break
            marker = data[idx + 1]
            if marker in (0xD9, 0xDA): break
            length = int.from_bytes(data[idx + 2:idx + 4], 'big')
            if marker == 0xE1:
                body = data[idx + 4 : idx + 2 + length]
                if body.startswith(b'Exif\x00\x00'):
                    tiff = body[6:]
                    endian = '<' if tiff[:2] == b'II' else ('>' if tiff[:2] == b'MM' else None)
                    if not endian: return 0
                    ifd = struct.unpack(endian + 'I', tiff[4:8])[0]
                    num = struct.unpack(endian + 'H', tiff[ifd : ifd + 2])[0]
                    if num <= 0 or num > 200 or ifd + 2 + num * 12 > len(tiff): return 0
                    for k in range(num):
                        ep = ifd + 2 + k * 12
                        if struct.unpack(endian + 'H', tiff[ep : ep + 2])[0] == 0x0112:
                            val = struct.unpack(endian + 'H', tiff[ep + 8 : ep + 10])[0]
                            if val in (1, 2, 3, 4, 6, 8): return val
                    return 0
            idx += 2 + length
    except Exception:
        pass
    return 0

def raw_embedded_orientation(file_path: str, embedded: bytes = None, light: bool = False) -> int:
    """RAW 方向唯一来源：ARW / CR2 等原生 TIFF 架构生肉强制以容器 IFD0 为准；CR3 优先内嵌预览自身 EXIF"""
    try:
        is_tiff_raw = file_path.lower().endswith(('.arw', '.cr2', '.nef', '.dng'))
        if embedded and not is_tiff_raw:
            val = _jpeg_exif_orientation(embedded)
            if val:
                return val
        return int(_raw_bundle(file_path, light=light).get("orientation", 1)) or 1
    except Exception:
        return 1

def load_photo_qimage(file_path: str, light: bool = False) -> QImage:
    """统一图片解码入口：RAW -> 提取内嵌 JPEG 预览（不解析 RAW）；普通图 -> QImageReader 原图解码。
    light=True 时 RAW 走轻量 bundle（缩略图批量遍历用，跳过 TIFF 枚举/EXIF/AF 派生）"""
    if file_path in _POISON_FILES and not is_raw_file(file_path):
        # 崩溃黑名单文件：不碰整图解码，退回 EXIF 小图
        try:
            tb, _ = extract_exif_thumb_and_orientation(file_path)
            if tb:
                return QImage.fromData(tb)
        except Exception:
            pass
        return QImage()
    if is_raw_file(file_path):
        buf = extract_raw_embedded_jpeg(file_path, light=light)
        if buf:
            img = QImage.fromData(buf)
            if not img.isNull():
                # 内嵌预览通常按横排存储，竖拍需要转正；
                # 方向与像素同源：直接读这段预览 JPEG 自己的 EXIF 方向
                orient = raw_embedded_orientation(file_path, buf, light=light)
                if orient == 6:
                    img = img.transformed(QTransform().rotate(90.0))
                elif orient == 8:
                    img = img.transformed(QTransform().rotate(-90.0))
                elif orient == 3:
                    img = img.transformed(QTransform().rotate(180.0))
                return img
        return QImage()
    reader = QImageReader(file_path)
    reader.setAutoTransform(True)
    return reader.read()

def load_shortcuts() -> dict:
    """读取当前生效的快捷键映射：动作 -> 键名(如 'A')"""
    settings = QSettings("AnicropGeeks", "AnicropStudio")
    return {a: str(settings.value(f"shortcuts/{a}", dflt)).upper() for a, (dflt, _, _) in DEFAULT_SHORTCUTS.items()}

def save_shortcut(action: str, key_name: str):
    settings = QSettings("AnicropGeeks", "AnicropStudio")
    settings.setValue(f"shortcuts/{action}", key_name.upper())

def reset_shortcuts():
    settings = QSettings("AnicropGeeks", "AnicropStudio")
    for a in DEFAULT_SHORTCUTS:
        settings.remove(f"shortcuts/{a}")

def shortcut_qt_key(key_name: str):
    """键名('A') -> Qt.Key 枚举值；非法返回 None"""
    return getattr(Qt, "Key_" + key_name.upper(), None)

def get_app_cache_root() -> Path:
    if getattr(sys, 'frozen', False):
        base_dir = Path(sys.executable).parent
    else:
        base_dir = Path(__file__).resolve().parent
    cache_root = base_dir / ".cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root

# ══════════════════════════════════════════════════════════════
# 崩溃黑匣子：PyInstaller 无窗口分发下 Python/Qt 的报错全被吞，
# 这里把 faulthandler(段错误级)、sys.excepthook(Python 异常)、Qt 警告三路全部落盘。
# 复现崩溃后把 .cache/crash.log 发回来即可定位死在哪一行。
# ══════════════════════════════════════════════════════════════
import faulthandler, traceback
_CRASH_LOG_FH = None

def _crashlog(msg: str):
    try:
        if _CRASH_LOG_FH:
            _CRASH_LOG_FH.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    except Exception:
        pass

_POISON_FILES = set()  # 上次运行中"开始解码却没完成"的文件（上次崩溃的凶手），本运行自动绕开

def _init_crash_blackbox():
    global _CRASH_LOG_FH
    try:
        log_path = get_app_cache_root() / "crash.log"
        # 启动时先翻上次日志：有"解码开始"无"解码完成"的文件就是上次的崩溃元凶，列入黑名单
        try:
            if log_path.exists():
                prev = log_path.read_text(encoding="utf-8", errors="ignore")
                started = re.findall(r"JPG全量解码开始 (.+)", prev)
                finished = set(re.findall(r"JPG全量解码完成 (.+?) null=", prev))
                _POISON_FILES.update(p.strip() for p in started if p.strip() not in finished)
                if _POISON_FILES:
                    pass  # 名单写进新日志，见下方启动行
        except Exception:
            pass
        # 只保留最近一次启动的日志，避免无限膨胀
        _CRASH_LOG_FH = open(log_path, "w", encoding="utf-8", buffering=1)
        faulthandler.enable(_CRASH_LOG_FH)
        _crashlog("=== 程序启动 ===")
        if _POISON_FILES:
            _crashlog(f"[blackbox] 上次崩溃疑似文件已拉黑 {len(_POISON_FILES)} 个: {sorted(_POISON_FILES)[:3]}")

        _orig_excepthook = sys.excepthook
        def _hook(exc_type, exc_val, exc_tb):
            try:
                _CRASH_LOG_FH.write("".join(traceback.format_exception(exc_type, exc_val, exc_tb)))
                _CRASH_LOG_FH.flush()
            except Exception:
                pass
            _orig_excepthook(exc_type, exc_val, exc_tb)
        sys.excepthook = _hook

        from PySide6.QtCore import qInstallMessageHandler, QtMsgType
        def _qt_msg(mode, ctx, msg):
            _crashlog(f"[Qt:{QtMsgType(mode).name}] {msg}")
        qInstallMessageHandler(_qt_msg)
    except Exception:
        pass

_init_crash_blackbox()

def _tiff_entries(tiff: bytes, endian: str, ifd_off: int) -> dict:
    """解析一个 IFD 的全部条目: tag -> (ftype, count, data_bytes)，已处理内联/偏移两种取值"""
    entries = {}
    if ifd_off <= 0 or ifd_off + 2 > len(tiff):
        return entries
    try:
        num = struct.unpack(endian + 'H', tiff[ifd_off : ifd_off + 2])[0]
        if num <= 0 or num > 200 or ifd_off + 2 + num * 12 > len(tiff):
            return entries
        for i in range(num):
            ep = ifd_off + 2 + i * 12
            tag, ftype, count = struct.unpack(endian + 'HHI', tiff[ep : ep + 8])
            raw4 = tiff[ep + 8 : ep + 12]
            unit = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 7: 1, 9: 4, 10: 8}.get(ftype, 1)
            total = unit * count
            if total <= 4:
                data_bytes = bytes(raw4[:total])
            else:
                off = struct.unpack(endian + 'I', raw4)[0]
                if off + total > len(tiff):
                    continue
                data_bytes = tiff[off : off + total]
            entries[tag] = (ftype, count, data_bytes)
    except Exception:
        pass
    return entries

def _ex_ascii(e) -> str:
    return e[2].split(b'\x00')[0].decode('utf-8', errors='ignore').strip()

def _ex_num(e, endian, signed=False):
    """short/long/rational/srational 第一值 -> float"""
    ftype, _, d = e
    try:
        if ftype == 3:
            return float(struct.unpack(endian + ('h' if signed else 'H'), d[:2])[0])
        if ftype in (4,):
            return float(struct.unpack(endian + ('i' if signed else 'I'), d[:4])[0])
        if ftype in (5, 10):
            num, den = struct.unpack(endian + ('ii' if signed else 'II'), d[:8])
            return num / den if den else None
        if ftype == 9:
            return float(struct.unpack(endian + 'i', d[:4])[0])
    except Exception:
        return None
    return None

def _fill_exif_from_tiff(res: dict, tiff: bytes, endian: str) -> bool:
    """从一块 TIFF 填 EXIF（只填仍是默认值的字段）。
    CR3 把 IFD0 / ExifIFD / MakerNote 拆成独立 uuid TIFF 块，需要跨块合并。
    返回本块是否贡献了新信息。
    """
    e0 = _tiff_entries(tiff, endian, struct.unpack(endian + 'I', tiff[4:8])[0])
    if not e0:
        return False
    sub = {}
    if 0x8769 in e0:
        sub = _tiff_entries(tiff, endian, struct.unpack(endian + 'I', e0[0x8769][2][:4])[0])

    # 曝光字段优先 ExifIFD，其次本块（CR3 的 ExifIFD 独立成块时字段就在本块）
    merged = dict(e0)
    merged.update(sub)
    if not any(t in merged for t in (0x010F, 0x0110, 0x829D, 0x8827, 0xA434)):
        return False  # 不像携带 EXIF 的块（缩略图 IFD 等），跳过

    contributed = False

    if res["camera"] == "未知机身":
        make = _ex_ascii(merged[0x010F]) if 0x010F in merged else ""
        model = _ex_ascii(merged[0x0110]) if 0x0110 in merged else ""
        if model:
            res["camera"] = model if (not make or model.lower().startswith(make.lower())) else f"{make} {model}"
            contributed = True
        elif make:
            res["camera"] = make
            contributed = True

    if res["lens"] == "未知镜头" and 0xA434 in merged:
        lens = _ex_ascii(merged[0xA434])
        if lens:
            res["lens"] = lens
            contributed = True

    # 【CR3 跨块合并护栏】CR3 部分 TIFF 块的值域偏移基准有私有 quirk，
    # 会解出 f/41486.1 之类的野值——物理量超界一律不采信，留给后续块补正确值
    if res["aperture"] == "--" and 0x829D in merged:
        v = _ex_num(merged[0x829D], endian)
        if v and 0.3 < v <= 128:
            res["aperture"] = f"f/{v:.1f}"
            contributed = True
    if res["shutter"] == "--" and 0x829A in merged:
        v = _ex_num(merged[0x829A], endian)
        if v and 0 < v < 36000:
            res["shutter"] = f"{v:.1f}s" if v >= 1 else f"1/{round(1.0 / v)}s"
            contributed = True
    if res["iso"] == "--" and 0x8827 in merged:
        v = _ex_num(merged[0x8827], endian)
        if v and 0 < v < 10000000:
            res["iso"] = f"ISO {int(v)}"
            contributed = True
    if res["ev"] == "--" and 0x9204 in merged:
        v = _ex_num(merged[0x9204], endian, signed=True)
        if v is not None and -30 < v < 30:
            res["ev"] = f"{v:+.1f} EV"
            contributed = True
    if res["focal"] == "--" and 0x920A in merged:
        v = _ex_num(merged[0x920A], endian)
        if v and 0 < v < 2000:
            res["focal"] = f"{v:.0f}mm"
            contributed = True

    if res["dimensions"] == "--" and 0xA002 in merged and 0xA003 in merged:
        w = _ex_num(merged[0xA002], endian)
        h = _ex_num(merged[0xA003], endian)
        if w and h:
            res["dimensions"] = f"{int(w)} × {int(h)}"
            contributed = True

    if res["date"] == "--":
        dt = _ex_ascii(merged[0x9003]) if 0x9003 in merged else (_ex_ascii(merged[0x0132]) if 0x0132 in merged else "")
        if dt:
            parts = dt.split(" ", 1)
            date_part = parts[0].replace(":", "-")
            res["date"] = (date_part + " " + parts[1]).strip() if len(parts) > 1 else date_part
            contributed = True

    return contributed

def _fill_exif_from_raw(res: dict, file_path: str):
    """RAW 容器 EXIF 填充：枚举全部 TIFF 候选跨块合并（走单趟缓存）"""
    try:
        ex = _raw_bundle(file_path).get("exif")
        if ex:
            for k, v in ex.items():
                if k in res and k != "size":
                    res[k] = v
    except Exception:
        pass

def parse_photo_exif(file_path: str) -> dict:
    """EXIF + XMP 双通道高保真参数提取器 (深度兼容像素蛋糕/Lightroom/PS导出图)"""
    res = {
        "camera": "未知机身",
        "lens": "未知镜头",
        "focal": "--",
        "aperture": "--",
        "shutter": "--",
        "iso": "--",
        "ev": "--",
        "date": "--",
        "size": "--",
        "dimensions": "--"
    }

    p = Path(file_path)
    if not p.exists():
        return res

    try:
        file_mb = p.stat().st_size / (1024 * 1024)
        res["size"] = f"{file_mb:.1f} MB"

        reader = QImageReader(file_path)
        sz = reader.size()
        if sz.isValid():
            res["dimensions"] = f"{sz.width()} × {sz.height()}"

        with open(file_path, 'rb') as f:
            header_data = f.read(524288)  # 读前 512KB，确保吃下全部元数据

        if len(header_data) < 4 or header_data[:2] != b'\xFF\xD8':
            if is_raw_file(file_path):
                _fill_exif_from_raw(res, file_path)
            return res

        # ---------------- 阶段 1: 扫描全部 APP1 段，提取标准 Exif TIFF ----------------
        idx = 2
        while idx < len(header_data) - 4:
            if header_data[idx] != 0xFF:
                break
            marker = header_data[idx + 1]
            if marker in (0xD9, 0xDA):
                break
            length = int.from_bytes(header_data[idx + 2:idx + 4], 'big')
            seg_end = idx + 2 + length
            if marker == 0xE1:
                body = header_data[idx + 4 : seg_end]
                if body.startswith(b'Exif\x00\x00'):
                    tiff = body[6:]
                    if len(tiff) >= 8:
                        endian_mark = tiff[:2]
                        endian = '<' if endian_mark == b'II' else ('>' if endian_mark == b'MM' else None)
                        if endian:
                            ifd0_offset = struct.unpack(endian + 'I', tiff[4:8])[0]
                            exif_sub_offset = None

                            if ifd0_offset + 2 <= len(tiff):
                                num_entries = struct.unpack(endian + 'H', tiff[ifd0_offset : ifd0_offset + 2])[0]
                                for i in range(min(num_entries, 100)):
                                    ep = ifd0_offset + 2 + i * 12
                                    if ep + 12 > len(tiff): break
                                    tag, ftype, count = struct.unpack(endian + 'HHI', tiff[ep : ep + 8])
                                    val_bytes = tiff[ep + 8 : ep + 12]

                                    if tag == 0x0110:
                                        str_offset = struct.unpack(endian + 'I', val_bytes)[0]
                                        if str_offset < len(tiff):
                                            raw_str = tiff[str_offset : str_offset + count].split(b'\x00')[0]
                                            res["camera"] = raw_str.decode('utf-8', errors='ignore').strip()
                                    elif tag == 0x8769:
                                        exif_sub_offset = struct.unpack(endian + 'I', val_bytes)[0]

                            if exif_sub_offset and exif_sub_offset + 2 <= len(tiff):
                                num_sub = struct.unpack(endian + 'H', tiff[exif_sub_offset : exif_sub_offset + 2])[0]
                                for i in range(min(num_sub, 200)):
                                    ep = exif_sub_offset + 2 + i * 12
                                    if ep + 12 > len(tiff): break
                                    tag, ftype, count = struct.unpack(endian + 'HHI', tiff[ep : ep + 8])
                                    val_bytes = tiff[ep + 8 : ep + 12]

                                    if tag == 0x829D:  # 光圈
                                        r_off = struct.unpack(endian + 'I', val_bytes)[0]
                                        if r_off + 8 <= len(tiff):
                                            num, den = struct.unpack(endian + 'II', tiff[r_off : r_off + 8])
                                            if den != 0: res["aperture"] = f"f/{num/den:.1f}"

                                    elif tag == 0x829A:  # 快门
                                        r_off = struct.unpack(endian + 'I', val_bytes)[0]
                                        if r_off + 8 <= len(tiff):
                                            num, den = struct.unpack(endian + 'II', tiff[r_off : r_off + 8])
                                            if den != 0:
                                                res["shutter"] = f"{num/den:.1f}s" if num >= den else f"1/{round(den/num)}s"

                                    elif tag in (0x8827, 0x8833):  # ISO (支持 SHORT 或 LONG)
                                        iso_val = struct.unpack(endian + ('I' if ftype == 4 else 'H'), val_bytes[:4 if ftype == 4 else 2])[0]
                                        if iso_val > 0: res["iso"] = f"ISO {iso_val}"

                                    elif tag == 0x920A:  # 焦距
                                        r_off = struct.unpack(endian + 'I', val_bytes)[0]
                                        if r_off + 8 <= len(tiff):
                                            num, den = struct.unpack(endian + 'II', tiff[r_off : r_off + 8])
                                            if den != 0: res["focal"] = f"{num/den:.0f}mm"

                                    elif tag == 0xA434 and ftype == 2:  # 镜头
                                        str_offset = struct.unpack(endian + 'I', val_bytes)[0]
                                        if str_offset < len(tiff):
                                            raw_str = tiff[str_offset : str_offset + count].split(b'\x00')[0]
                                            res["lens"] = raw_str.decode('utf-8', errors='ignore').strip()

                                    elif tag == 0x9003 and ftype == 2:  # 拍摄时间
                                        str_offset = struct.unpack(endian + 'I', val_bytes)[0]
                                        if str_offset < len(tiff):
                                            raw_str = tiff[str_offset : str_offset + count].split(b'\x00')[0]
                                            dt_str = raw_str.decode('utf-8', errors='ignore').strip()
                                            if len(dt_str) >= 10:
                                                dt_str = dt_str[:10].replace(":", ".") + dt_str[10:]
                                            res["date"] = dt_str
            idx = seg_end

        # ---------------- 阶段 2: XMP 深度探测兜底 (像素蛋糕 / PS / LR 核心防线) ----------------
        if res["camera"] == "未知机身" or res["lens"] == "未知镜头" or res["aperture"] == "--":
            xmp_idx = header_data.find(b'<x:xmpmeta')
            if xmp_idx >= 0:
                xmp_chunk = header_data[xmp_idx : xmp_idx + 131072]

                def _xmp_find(patterns):
                    for pat in patterns:
                        m = re.search(pat, xmp_chunk)
                        if m:
                            try:
                                return m.group(1).decode('utf-8', 'ignore').strip()
                            except Exception:
                                pass
                    return None

                # 提取机身
                if res["camera"] == "未知机身":
                    cam = _xmp_find([rb'<tiff:Model>([^<]+)</tiff:Model>', rb'tiff:Model="([^"]+)"'])
                    if cam: res["camera"] = cam

                # 提取镜头
                if res["lens"] == "未知镜头":
                    lens = _xmp_find([rb'<aux:Lens>([^<]+)</aux:Lens>', rb'aux:Lens="([^"]+)"',
                                     rb'<exifEX:LensModel>([^<]+)</exifEX:LensModel>', rb'exifEX:LensModel="([^"]+)"'])
                    if lens: res["lens"] = lens

                # 提取光圈
                if res["aperture"] == "--":
                    ap = _xmp_find([rb'<exif:FNumber>([^<]+)</exif:FNumber>', rb'exif:FNumber="([^"]+)"'])
                    if ap:
                        try:
                            v = eval(ap) if '/' in ap else float(ap)
                            res["aperture"] = f"f/{v:.1f}"
                        except Exception:
                            pass

                # 提取快门
                if res["shutter"] == "--":
                    sh = _xmp_find([rb'<exif:ExposureTime>([^<]+)</exif:ExposureTime>', rb'exif:ExposureTime="([^"]+)"'])
                    if sh:
                        try:
                            if '/' in sh:
                                res["shutter"] = f"{sh}s"
                            else:
                                fv = float(sh)
                                res["shutter"] = f"{fv:.1f}s" if fv >= 1 else f"1/{round(1.0/fv)}s"
                        except Exception:
                            pass

                # 提取 ISO
                if res["iso"] == "--":
                    iso = _xmp_find([rb'<exif:ISOSpeedRatings>[^<]*<rdf:li>(\d+)</rdf:li>', rb'<exif:PhotographicSensitivity>(\d+)</exif:PhotographicSensitivity>'])
                    if iso: res["iso"] = f"ISO {iso}"

                # 提取焦距
                if res["focal"] == "--":
                    foc = _xmp_find([rb'<exif:FocalLength>([^<]+)</exif:FocalLength>', rb'exif:FocalLength="([^"]+)"'])
                    if foc:
                        try:
                            v = eval(foc) if '/' in foc else float(foc)
                            res["focal"] = f"{round(v)}mm"
                        except Exception:
                            pass

                # 提取时间
                if res["date"] == "--":
                    dt = _xmp_find([rb'<exif:DateTimeOriginal>([^<]+)</exif:DateTimeOriginal>', rb'exif:DateTimeOriginal="([^"]+)"'])
                    if dt:
                        res["date"] = dt.replace("T", " ")[:19].replace("-", ".")
    except Exception:
        pass

    return res

    try:
        file_mb = p.stat().st_size / (1024 * 1024)
        res["size"] = f"{file_mb:.1f} MB"

        reader = QImageReader(file_path)
        sz = reader.size()
        if sz.isValid():
            res["dimensions"] = f"{sz.width()} × {sz.height()}"

        with open(file_path, 'rb') as f:
            header_data = f.read(131072)

        if len(header_data) < 4 or header_data[:2] != b'\xFF\xD8':
            if is_raw_file(file_path):
                _fill_exif_from_raw(res, file_path)
            return res

        idx = 2
        while idx < len(header_data) - 4:
            if header_data[idx] != 0xFF:
                break
            marker = header_data[idx + 1]
            if marker in (0xD9, 0xDA):
                break
            length = int.from_bytes(header_data[idx + 2:idx + 4], 'big')
            if marker == 0xE1:
                exif_body = header_data[idx + 4 : idx + 2 + length]
                if exif_body.startswith(b'Exif\x00\x00'):
                    tiff = exif_body[6:]
                    if len(tiff) >= 8:
                        endian_mark = tiff[:2]
                        endian = '<' if endian_mark == b'II' else ('>' if endian_mark == b'MM' else None)
                        if endian:
                            ifd0_offset = struct.unpack(endian + 'I', tiff[4:8])[0]
                            exif_sub_offset = None

                            if ifd0_offset + 2 <= len(tiff):
                                num_entries = struct.unpack(endian + 'H', tiff[ifd0_offset : ifd0_offset + 2])[0]
                                for i in range(num_entries):
                                    ep = ifd0_offset + 2 + i * 12
                                    if ep + 12 > len(tiff): break
                                    tag, ftype, count = struct.unpack(endian + 'HHI', tiff[ep : ep + 8])
                                    val_bytes = tiff[ep + 8 : ep + 12]

                                    if tag == 0x0110:
                                        str_offset = struct.unpack(endian + 'I', val_bytes)[0]
                                        if str_offset < len(tiff):
                                            raw_str = tiff[str_offset : str_offset + count].split(b'\x00')[0]
                                            res["camera"] = raw_str.decode('utf-8', errors='ignore').strip()
                                    elif tag == 0x8769:
                                        exif_sub_offset = struct.unpack(endian + 'I', val_bytes)[0]

                            if exif_sub_offset and exif_sub_offset + 2 <= len(tiff):
                                num_sub = struct.unpack(endian + 'H', tiff[exif_sub_offset : exif_sub_offset + 2])[0]
                                for i in range(num_sub):
                                    ep = exif_sub_offset + 2 + i * 12
                                    if ep + 12 > len(tiff): break
                                    tag, ftype, count = struct.unpack(endian + 'HHI', tiff[ep : ep + 8])
                                    val_bytes = tiff[ep + 8 : ep + 12]

                                    if tag == 0x829D and ftype == 5:
                                        r_off = struct.unpack(endian + 'I', val_bytes)[0]
                                        if r_off + 8 <= len(tiff):
                                            num, den = struct.unpack(endian + 'II', tiff[r_off : r_off + 8])
                                            if den != 0: res["aperture"] = f"f/{num/den:.1f}"

                                    elif tag == 0x829A and ftype == 5:
                                        r_off = struct.unpack(endian + 'I', val_bytes)[0]
                                        if r_off + 8 <= len(tiff):
                                            num, den = struct.unpack(endian + 'II', tiff[r_off : r_off + 8])
                                            if den != 0:
                                                if num >= den:
                                                    res["shutter"] = f"{num/den:.1f}s"
                                                else:
                                                    res["shutter"] = f"1/{round(den/num)}s"

                                    elif tag == 0x8827:
                                        iso_val = struct.unpack(endian + 'H', val_bytes[:2])[0]
                                        res["iso"] = f"ISO {iso_val}"

                                    elif tag == 0x920A and ftype == 5:
                                        r_off = struct.unpack(endian + 'I', val_bytes)[0]
                                        if r_off + 8 <= len(tiff):
                                            num, den = struct.unpack(endian + 'II', tiff[r_off : r_off + 8])
                                            if den != 0: res["focal"] = f"{num/den:.0f}mm"

                                    elif tag == 0xA434 and ftype == 2:
                                        str_offset = struct.unpack(endian + 'I', val_bytes)[0]
                                        if str_offset < len(tiff):
                                            raw_str = tiff[str_offset : str_offset + count].split(b'\x00')[0]
                                            res["lens"] = raw_str.decode('utf-8', errors='ignore').strip()

                                    elif tag == 0x9003 and ftype == 2:
                                        str_offset = struct.unpack(endian + 'I', val_bytes)[0]
                                        if str_offset < len(tiff):
                                            raw_str = tiff[str_offset : str_offset + count].split(b'\x00')[0]
                                            dt_str = raw_str.decode('utf-8', errors='ignore').strip()
                                            if len(dt_str) >= 10:
                                                dt_str = dt_str[:10].replace(":", "-") + dt_str[10:]
                                            res["date"] = dt_str

                                    elif tag == 0x9204 and ftype == 10:
                                        r_off = struct.unpack(endian + 'I', val_bytes)[0]
                                        if r_off + 8 <= len(tiff):
                                            num, den = struct.unpack(endian + 'ii', tiff[r_off : r_off + 8])
                                            if den != 0:
                                                ev_val = num / den
                                                res["ev"] = f"{ev_val:+.1f} EV"
                break
            idx += 2 + length
    except Exception:
        pass

    return res

def parse_photo_rating(file_path: str) -> int:
    """
    读取照片机内打星 (xmp:Rating 0~5 / Sony MakerNote 0x2002)，只读不写。
    兼容: JPEG / 佳能 CR3·CR2 / 索尼 ARW / 尼康 NEF 及同名 .xmp 伴侣文件。
    未打星或解析失败一律返回 0。
    """
    def _rating_from(buf: bytes) -> int:
        m = re.search(rb'xmp:Rating\s*=\s*"\s*(-?\d)\s*"', buf)
        if not m:
            m = re.search(rb'<xmp:Rating>\s*(-?\d)\s*</xmp:Rating>', buf)
        if not m:
            return 0
        try:
            return max(0, min(5, int(m.group(1))))
        except Exception:
            return 0

    p = Path(file_path)
    try:
        # 1. 优先嗅探同名 .xmp 伴侣文件 (Lightroom / Bridge / 达芬奇)
        for sidecar in (p.with_suffix('.xmp'), p.with_suffix('.XMP'), p.parent / (p.name + '.xmp')):
            if sidecar.exists():
                try:
                    s_val = _rating_from(sidecar.read_bytes())
                    if s_val > 0:
                        return s_val
                except Exception:
                    pass

        ext = p.suffix.lower()
        if ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
            with open(p, 'rb') as f:
                head = f.read(262144)
            idx = head.find(b'<x:xmpmeta')
            if idx >= 0:
                return _rating_from(head[idx: idx + 65536])
            return _rating_from(head)

        elif ext in (".cr3", ".cr2", ".arw", ".nef", ".dng"):
            ent = _RAW_BUNDLE_CACHE.get(file_path)
            if ent is not None:
                try:
                    return int(ent.get("rating", 0))
                except Exception:
                    return 0

            with open(p, 'rb') as f:
                head_chunk = f.read(4 * 1024 * 1024)

            # 内嵌 XMP 扫描
            idx = head_chunk.find(b'<x:xmpmeta')
            if idx >= 0:
                r = _rating_from(head_chunk[idx: idx + 65536])
                if r > 0:
                    return r

            # 索尼 ARW 专属：MakerNote 0x2002 硬件机内打星
            if ext == '.arw':
                pos = 0
                while True:
                    idx_sony = head_chunk.find(b'SONY DSC', pos)
                    if idx_sony == -1:
                        break
                    if idx_sony + 14 <= len(head_chunk):
                        cnt = struct.unpack_from('<H', head_chunk, idx_sony + 12)[0]
                        if 0 < cnt <= 512:
                            for i in range(cnt):
                                ep = idx_sony + 14 + i * 12
                                if ep + 12 > len(head_chunk):
                                    break
                                tag = struct.unpack_from('<H', head_chunk, ep)[0]
                                if tag == 0x2002:
                                    r_val = struct.unpack_from('<I', head_chunk, ep + 8)[0]
                                    if 1 <= r_val <= 5:
                                        return r_val
                    pos = idx_sony + 8

            return 0
    except Exception:
        return 0
    return 0

def extract_exif_thumb_and_orientation(file_path: str):
    """从 JPEG 文件头提取 EXIF 内嵌缩略图(通常 160×120)与方向。只读文件头 + 缩略图本体，不整读文件"""
    orientation = 1
    thumb_bytes = b""
    try:
        with open(file_path, 'rb') as f:
            header_data = f.read(131072)

        if len(header_data) < 4 or header_data[:2] != b'\xFF\xD8':
            return b"", 1

        idx = 2
        while idx < len(header_data) - 4:
            if header_data[idx] != 0xFF:
                break
            marker = header_data[idx + 1]
            if marker in (0xD9, 0xDA):
                break
            length = int.from_bytes(header_data[idx + 2:idx + 4], 'big')
            if marker == 0xE1:
                exif_body = header_data[idx + 4 : idx + 2 + length]
                if exif_body.startswith(b'Exif\x00\x00'):
                    tiff = exif_body[6:]
                    if len(tiff) >= 8:
                        endian_mark = tiff[:2]
                        endian = '<' if endian_mark == b'II' else ('>' if endian_mark == b'MM' else None)
                        if endian:
                            ifd0_offset = struct.unpack(endian + 'I', tiff[4:8])[0]
                            if ifd0_offset + 2 <= len(tiff):
                                num_entries = struct.unpack(endian + 'H', tiff[ifd0_offset : ifd0_offset + 2])[0]
                                thumb_ifd_off = 0
                                for i in range(num_entries):
                                    entry_pos = ifd0_offset + 2 + i * 12
                                    if entry_pos + 12 > len(tiff): break
                                    tag = struct.unpack(endian + 'H', tiff[entry_pos : entry_pos + 2])[0]
                                    if tag == 0x0112:
                                        orientation = struct.unpack(endian + 'H', tiff[entry_pos + 8 : entry_pos + 10])[0]
                                # IFD0 末尾的 next-IFD 指针指向 IFD1（内嵌缩略图）
                                nxt = ifd0_offset + 2 + num_entries * 12
                                if nxt + 4 <= len(tiff):
                                    thumb_ifd_off = struct.unpack(endian + 'I', tiff[nxt : nxt + 4])[0]
                                if 0 < thumb_ifd_off and thumb_ifd_off + 2 <= len(tiff):
                                    n1 = struct.unpack(endian + 'H', tiff[thumb_ifd_off : thumb_ifd_off + 2])[0]
                                    t_off = t_len = 0
                                    for i in range(min(n1, 200)):
                                        ep = thumb_ifd_off + 2 + i * 12
                                        if ep + 12 > len(tiff): break
                                        tag = struct.unpack(endian + 'H', tiff[ep : ep + 2])[0]
                                        if tag == 0x0201:      # JPEGInterchangeFormat
                                            t_off = struct.unpack(endian + 'I', tiff[ep + 8 : ep + 12])[0]
                                        elif tag == 0x0202:    # JPEGInterchangeFormatLength
                                            t_len = struct.unpack(endian + 'I', tiff[ep + 8 : ep + 12])[0]
                                    if t_off > 0 and 64 < t_len <= 1024 * 1024:
                                        # 缩略图本体通常在 128KB 头内；不在则按偏移补读一小段
                                        if t_off + t_len <= len(tiff):
                                            cand = tiff[t_off : t_off + t_len]
                                        else:
                                            app1_data_off = idx + 4 + 6  # tiff 起点在文件中的绝对偏移
                                            with open(file_path, 'rb') as f2:
                                                f2.seek(app1_data_off + t_off)
                                                cand = f2.read(t_len)
                                        if cand[:2] == b'\xFF\xD8':
                                            thumb_bytes = cand
                break
            idx += 2 + length
    except Exception:
        pass
    return thumb_bytes, orientation

def _iter_raw_tiff_candidates(data: bytes):
    """枚举 RAW 容器头部全部可疑 TIFF 块并校验 IFD 结构。
    TIFF 系 (CR2/NEF/ARW/DNG) 文件头即 TIFF；CR3 (BMFF) 的 EXIF/CMT uuid 盒内嵌 TIFF。
    返回 [(tiff_bytes, endian), ...]，可能多块，调用方逐块尝试。
    """
    results = []
    if len(data) < 16:
        return results

    raw_offsets = []
    if data[:4] in (b'II*\x00', b'MM\x00*'):
        raw_offsets.append(0)
    pos = 0
    while len(raw_offsets) < 128:
        k = data.find(b'Exif\x00\x00', pos)
        if k < 0:
            break
        raw_offsets.append(k + 6)
        pos = k + 6
    # TIFF 魔数全枚举（CR3 的 CMT1/CMT2 等 uuid 盒 payload 裸 TIFF 就靠这个兜住）
    pos = 0
    while len(raw_offsets) < 256:
        i1 = data.find(b'II*\x00', pos)
        i2 = data.find(b'MM\x00*', pos)
        picks = [x for x in (i1, i2) if x >= 0]
        if not picks:
            break
        raw_offsets.append(min(picks))
        pos = min(picks) + 4

    seen = set()
    for off in raw_offsets:
        if off in seen or off + 8 > len(data):
            continue
        seen.add(off)
        tiff = data[off:]
        if tiff[:4] not in (b'II*\x00', b'MM\x00*'):
            continue
        endian = '<' if tiff[:2] == b'II' else '>'
        try:
            ifd0 = struct.unpack(endian + 'I', tiff[4:8])[0]
            if ifd0 > 8 * 1024 * 1024 or ifd0 + 2 > len(tiff):
                continue
            num = struct.unpack(endian + 'H', tiff[ifd0 : ifd0 + 2])[0]
            if 0 < num <= 200 and ifd0 + 2 + num * 12 <= len(tiff):
                results.append((tiff, endian))
        except Exception:
            continue
    return results

def extract_raw_orientation(file_path: str) -> int:
    """读取 RAW 容器内 EXIF 方向（走单趟缓存）"""
    try:
        return int(_raw_bundle(file_path).get("orientation", 1)) or 1
    except Exception:
        return 1

def _exif_tiff_block(data: bytes) -> bytes | None:
    """从 JPEG 字节流中定位并提取 APP1/Exif 后的 TIFF 数据块"""
    if len(data) < 4 or data[:2] != b'\xFF\xD8':
        return None
    idx = 2
    n = len(data)
    while idx < n - 4:
        if data[idx] != 0xFF:
            break
        m = data[idx + 1]
        if m in (0xD9, 0xDA):
            break
        if m == 0xD8 or m == 0x01 or 0xD0 <= m <= 0xD7:
            idx += 2
            continue
        ln = int.from_bytes(data[idx + 2:idx + 4], 'big')
        if m == 0xE1:
            body = data[idx + 4:idx + 2 + ln]
            if body.startswith(b'Exif\x00\x00'):
                tiff = body[6:]
                if tiff[:2] in (b'II', b'MM'):
                    return tiff
        idx += 2 + ln
    return None


def _makernote_vendor(tiff_data: bytes) -> bytes | None:
    """快速嗅探 MakerNote 厂商签名：b'SONY' / b'CANON' 等"""
    if len(tiff_data) < 8:
        return None
    endian = '<' if tiff_data[:2] == b'II' else ('>' if tiff_data[:2] == b'MM' else None)
    if not endian:
        return None
    try:
        ifd0_off = struct.unpack(endian + 'I', tiff_data[4:8])[0]
        if ifd0_off + 2 > len(tiff_data):
            return None
        num_entries = struct.unpack(endian + 'H', tiff_data[ifd0_off:ifd0_off + 2])[0]
        mn_off = None
        exif_off = None
        for i in range(num_entries):
            ep = ifd0_off + 2 + i * 12
            if ep + 12 > len(tiff_data):
                break
            tag = struct.unpack(endian + 'H', tiff_data[ep:ep + 2])[0]
            if tag == 0x927C:
                mn_off = struct.unpack(endian + 'I', tiff_data[ep + 8:ep + 12])[0]
                break
            elif tag == 0x8769:
                exif_off = struct.unpack(endian + 'I', tiff_data[ep + 8:ep + 12])[0]
        if mn_off is None and exif_off and exif_off + 2 <= len(tiff_data):
            n2 = struct.unpack(endian + 'H', tiff_data[exif_off:exif_off + 2])[0]
            for j in range(n2):
                ep = exif_off + 2 + j * 12
                if ep + 12 > len(tiff_data):
                    break
                tag = struct.unpack(endian + 'H', tiff_data[ep:ep + 2])[0]
                if tag == 0x927C:
                    mn_off = struct.unpack(endian + 'I', tiff_data[ep + 8:ep + 12])[0]
                    break
        if mn_off and mn_off + 12 <= len(tiff_data):
            head = tiff_data[mn_off:mn_off + 12]
            if head.startswith(b'SONY'):
                return b'SONY'
            if head.startswith(b'Nikon'):
                return b'NIKON'
    except Exception:
        pass
    return None


def _af_from_jpeg_bytes(data: bytes, orientation: int = 1) -> list:
    """从 JPEG 字节流按厂商分发解析 AF 框"""
    try:
        tiff_data = _exif_tiff_block(data)
        if tiff_data is None:
            return []
        vendor = _makernote_vendor(tiff_data)
        if vendor == b'SONY':
            boxes = _sony_af_core(tiff_data, '<', orientation)
            if boxes:
                return boxes
        elif vendor == b'NIKON':
            boxes = _nikon_af_core(tiff_data, '<', orientation)
            if boxes:
                return boxes
        return _canon_af_core(tiff_data, '<' if tiff_data[:2] == b'II' else '>', orientation) or []
    except Exception:
        return []


def extract_af_points(file_path: str, orientation: int = 1) -> list:
    """【品牌无关】AF 对焦点提取总入口：支持佳能 EOS R / 索尼 α 系列。"""
    try:
        is_raw = is_raw_file(file_path)
        if is_raw:
            return list(_raw_bundle(file_path).get("af") or [])

        with open(file_path, 'rb') as f:
            data = f.read(250000)
        return _af_from_jpeg_bytes(data, orientation)
    except Exception:
        return []


# 【核心兼容垫片】防止旧代码 / debug_focus.py 依赖原函数名报错
extract_canon_af_points = extract_af_points

def _canon_af_core(tiff_data: bytes, endian: str, orientation: int = 1) -> list:
    af_boxes = []
    try:
        ifd0_off = struct.unpack(endian + 'I', tiff_data[4:8])[0]
        makernote_off = None
        exif_sub_off = None

        num_entries = struct.unpack(endian + 'H', tiff_data[ifd0_off : ifd0_off + 2])[0]
        for i in range(num_entries):
            ep = ifd0_off + 2 + i * 12
            tag = struct.unpack(endian + 'H', tiff_data[ep : ep + 2])[0]
            val_bytes = tiff_data[ep + 8 : ep + 12]
            if tag == 0x927C: makernote_off = struct.unpack(endian + 'I', val_bytes)[0]
            elif tag == 0x8769: exif_sub_off = struct.unpack(endian + 'I', val_bytes)[0]

        if not makernote_off and exif_sub_off:
            num_sub = struct.unpack(endian + 'H', tiff_data[exif_sub_off : exif_sub_off + 2])[0]
            for i in range(num_sub):
                ep = exif_sub_off + 2 + i * 12
                tag = struct.unpack(endian + 'H', tiff_data[ep : ep + 2])[0]
                if tag == 0x927C:
                    makernote_off = struct.unpack(endian + 'I', tiff_data[ep + 8 : ep + 12])[0]
                    break

        af_data_bytes = None

        if makernote_off:
            # JPEG/CR2 经典路径：IFD0 或 ExifIFD 里的 0x927C 指向 MakerNote IFD
            mn_entries = struct.unpack(endian + 'H', tiff_data[makernote_off : makernote_off + 2])[0]
            for i in range(mn_entries):
                ep = makernote_off + 2 + i * 12
                if ep + 12 > len(tiff_data): break
                tag, ftype, count = struct.unpack(endian + 'HHI', tiff_data[ep : ep + 8])
                if tag in (0x0026, 0x003C):
                    off = struct.unpack(endian + 'I', tiff_data[ep + 8 : ep + 12])[0]
                    total_bytes = count * 2 if ftype == 3 else count * 4
                    af_data_bytes = tiff_data[off : off + total_bytes]
                    break
        else:
            # 【CR3 特例】佳能 CR3 把 MakerNote 存成独立 uuid TIFF 块（无 0x927C 指针可挂）：
            # 本块 IFD0 若直接挂 AFInfo (0x0026/0x003C)，就把本块当 MakerNote 用
            e0 = _tiff_entries(tiff_data, endian, struct.unpack(endian + 'I', tiff_data[4:8])[0])
            for af_tag in (0x0026, 0x003C):
                if af_tag in e0:
                    af_data_bytes = e0[af_tag][2]
                    break

        if not af_data_bytes: return []

        shorts = [struct.unpack(endian + 'h', af_data_bytes[j:j+2])[0] for j in range(0, len(af_data_bytes), 2)]
        ushorts = [struct.unpack(endian + 'H', af_data_bytes[j:j+2])[0] for j in range(0, len(af_data_bytes), 2)]

        num_points = shorts[2]
        af_w = float(shorts[6])
        af_h = float(shorts[7])

        if num_points <= 0 or af_w <= 0 or af_h <= 0 or len(shorts) < (8 + num_points * 4):
            return []

        widths = shorts[8 : 8 + num_points]
        heights = shorts[8 + num_points : 8 + num_points * 2]
        xs = shorts[8 + num_points * 2 : 8 + num_points * 3]
        ys = shorts[8 + num_points * 3 : 8 + num_points * 4]
        mask_start = 8 + num_points * 4
        primary_idx = shorts[-1]

        in_focus_set = set()
        for k in range(num_points):
            w_idx = mask_start + (k // 16)
            if w_idx < len(ushorts):
                if ((ushorts[w_idx] >> (k % 16)) & 1) == 1:
                    in_focus_set.add(k)
        if primary_idx >= 0:
            in_focus_set.add(primary_idx)

        for k in range(num_points):
            is_f = k in in_focus_set
            if is_f:
                bw = widths[k] if widths[k] > 0 else 262.0
                bh = heights[k] if heights[k] > 0 else 262.0

                cx = (af_w / 2.0) + xs[k]
                cy = (af_h / 2.0) - ys[k]

                norm_cx = cx / af_w
                norm_cy = cy / af_h
                norm_w = bw / af_w
                norm_h = bh / af_h

                if orientation == 6:
                    rot_cx, rot_cy = 1.0 - norm_cy, norm_cx
                    rot_w, rot_h = norm_h, norm_w
                elif orientation == 8:
                    rot_cx, rot_cy = norm_cy, 1.0 - norm_cx
                    rot_w, rot_h = norm_h, norm_w
                elif orientation == 3:
                    rot_cx, rot_cy = 1.0 - norm_cx, 1.0 - norm_cy
                    rot_w, rot_h = norm_w, norm_h
                else:
                    rot_cx, rot_cy = norm_cx, norm_cy
                    rot_w, rot_h = norm_w, norm_h

                rx = max(0.0, rot_cx - rot_w / 2.0)
                ry = max(0.0, rot_cy - rot_h / 2.0)
                af_boxes.append({
                    "rect": QRectF(rx, ry, rot_w, rot_h),
                    "in_focus": is_f
                })
    except Exception:
        pass

    return af_boxes


class ThumbnailSignals(QObject):
    # 【线程安全】worker 线程严禁创建 QPixmap（GUI 线程专属，子线程碰它会随机闪退）——
    # 一律产 QImage 跨线程传递，槽函数（GUI 线程）里再转 QPixmap
    loaded = Signal(str, QImage)
    task_done = Signal(str)
    meta_loaded = Signal(str, int)   # (路径, 打星 0~5)
    provisional_used = Signal(str)   # (路径) 预热档用了 EXIF 小图/未出图，需要升级梯队补高清

class ThumbnailTask(QRunnable):
    def __init__(self, file_path: str, target_cache_dir: Path, target_size: int = 512, provisional_ok: bool = False):
        super().__init__()
        self.file_path = file_path
        self.target_cache_dir = target_cache_dir
        self.target_size = target_size
        self.provisional_ok = provisional_ok  # True=预热档：JPG 允许只用 EXIF 小图快速出图，高清图留给升级梯队
        self.signals = ThumbnailSignals()

    def run(self):
        # RAW 缩略图批量遍历：先预热轻量 bundle（一趟读盘，只抽预览/星级，跳过 TIFF 枚举/EXIF/AF 派生），
        # 随后的 parse_photo_rating / load_photo_qimage 全部命中同一缓存条目
        if is_raw_file(self.file_path):
            try:
                _raw_bundle(self.file_path, light=True)
            except Exception:
                pass
        try:
            self.signals.meta_loaded.emit(self.file_path, parse_photo_rating(self.file_path))
        except Exception:
            pass
        try:
            mtime = os.path.getmtime(self.file_path)
            hash_str = f"{self.file_path}_{mtime}"
            cache_key = hashlib.md5(hash_str.encode("utf-8")).hexdigest()
            cache_file = self.target_cache_dir / f"{cache_key}.jpg"

            if cache_file.exists():
                img = QImage(str(cache_file))
                if not img.isNull():
                    self.signals.loaded.emit(self.file_path, img)
                    return  # task_done 由 finally 统一发，切勿重复（会冲爆预热计数）

            if is_raw_file(self.file_path):
                # RAW：抽内嵌预览再缩略（不解析 RAW；轻量 bundle，跳过 EXIF/AF 派生）
                raw_img = load_photo_qimage(self.file_path, light=True)
                img = raw_img.scaled(self.target_size, self.target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation) if not raw_img.isNull() else QImage()
            elif self.provisional_ok:
                # JPG 预热档：只读文件头抠 EXIF 内嵌小图（~128KB/张），整图解码留给升级梯队。
                # 无论小图成败都挂进升级队列（没有小图的交给升级梯队全量解码）
                img = None
                try:
                    tb, tor = extract_exif_thumb_and_orientation(self.file_path)
                    if tb:
                        timg = QImage.fromData(tb)
                        if not timg.isNull():
                            if tor == 6:
                                timg = timg.transformed(QTransform().rotate(90.0))
                            elif tor == 8:
                                timg = timg.transformed(QTransform().rotate(-90.0))
                            elif tor == 3:
                                timg = timg.transformed(QTransform().rotate(180.0))
                            timg = timg.scaled(self.target_size, self.target_size, Qt.KeepAspectRatio, Qt.FastTransformation)
                            self.signals.loaded.emit(self.file_path, timg)
                except Exception:
                    pass
                try:
                    self.signals.provisional_used.emit(self.file_path)
                except Exception:
                    pass
                return  # task_done 由 finally 统一发，切勿重复（会冲爆预热计数）
            elif self.file_path in _POISON_FILES:
                # 上次崩溃的疑似凶手文件：跳过全量解码，退回 EXIF 小图兜底
                _crashlog(f"[thumb] 跳过黑名单文件 {self.file_path}")
                try:
                    tb, tor = extract_exif_thumb_and_orientation(self.file_path)
                    if tb:
                        timg = QImage.fromData(tb)
                        if not timg.isNull():
                            if tor == 6:
                                timg = timg.transformed(QTransform().rotate(90.0))
                            elif tor == 8:
                                timg = timg.transformed(QTransform().rotate(-90.0))
                            elif tor == 3:
                                timg = timg.transformed(QTransform().rotate(180.0))
                            self.signals.loaded.emit(self.file_path, timg.scaled(
                                self.target_size, self.target_size, Qt.KeepAspectRatio, Qt.FastTransformation))
                except Exception:
                    pass
                return
            else:
                _crashlog(f"[thumb] JPG全量解码开始 {self.file_path}")
                reader = QImageReader(self.file_path)
                reader.setAutoTransform(True)
                orig_size = reader.size()
                if orig_size.isValid() and orig_size.width() > 0 and orig_size.height() > 0:
                    scaled_size = orig_size.scaled(self.target_size, self.target_size, Qt.KeepAspectRatio)
                    reader.setScaledSize(scaled_size)
                img = reader.read()
                _crashlog(f"[thumb] JPG全量解码完成 {self.file_path} null={img.isNull()}")
            if not img.isNull():
                try:
                    img.save(str(cache_file), "JPG", 85)
                except Exception:
                    pass
                self.signals.loaded.emit(self.file_path, img)
        except Exception:
            pass
        finally:
            self.signals.task_done.emit(self.file_path)

class PhotoListModel(QAbstractListModel):
    ROLE_IS_DIR = Qt.UserRole + 1

    def __init__(self):
        super().__init__()
        self.items = []

    def rowCount(self, parent=QModelIndex()):
        return len(self.items)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < len(self.items)):
            return None
        path, is_dir = self.items[index.row()]
        if role == Qt.UserRole:
            return path
        elif role == self.ROLE_IS_DIR:
            return is_dir
        elif role == Qt.DisplayRole:
            return Path(path).name
        return None

    def set_items(self, item_tuples):
        self.beginResetModel()
        self.items = list(item_tuples)
        self.endResetModel()

    def set_photos(self, photo_paths):
        self.set_items([(p, False) for p in photo_paths])

    def get_item(self, row: int):
        if 0 <= row < len(self.items):
            return self.items[row]
        return "", False

class AdvancedCardDelegate(QStyledItemDelegate):
    def __init__(self, parent=None, is_darkroom=False, cull_mode="reject"):
        super().__init__(parent)
        self.is_darkroom = is_darkroom
        self.cull_mode = cull_mode
        self.card_size = QSize(220, 250)
        self.thumb_cache = {}
        self.ratings = {}
        self.offset_x = 0  # 动态对称居中偏移量
        
        self.is_selection_mode = False
        self.selected_set = set()
        self.hidden_set = set()
        self.is_remove_mode = False
        self.remove_set = set()

    def set_card_size(self, size: QSize):
        self.card_size = size

    def clear_cache(self):
        self.thumb_cache.clear()

    def sizeHint(self, option, index):
        return self.card_size

    def paint(self, painter: QPainter, option, index):
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)

        # 加上 offset_x，整排卡片向右平移居中，彻底消除右侧大黑块！
        rect = option.rect.adjusted(4 + self.offset_x, 4, -4 + self.offset_x, -4)
        file_path = index.data(Qt.UserRole)
        file_name = index.data(Qt.DisplayRole)
        is_dir = bool(index.data(PhotoListModel.ROLE_IS_DIR))

        is_item_selected = bool(option.state & QStyle.State_Selected)
        is_hover = bool(option.state & QStyle.State_MouseOver)
        is_checked = (file_path in self.selected_set) and not is_dir
        is_hidden = (file_path in self.hidden_set) and not is_dir

        card_bg = QColor("#1e1e24")
        border_color = QColor("#27272a")

        if is_item_selected:
            card_bg = QColor("#172554")
            border_color = QColor("#38bdf8")
        elif is_checked:
            card_bg = QColor("#1e293b")
            border_color = QColor("#0284c7")
        elif is_hover:
            border_color = QColor("#52525b")

        if is_hidden:
            card_bg = QColor("#121215")
            border_color = QColor("#222226")

        painter.setBrush(QBrush(card_bg))
        painter.setPen(QPen(border_color, 1.8 if (is_item_selected or is_checked) else 1.0))
        painter.drawRoundedRect(rect, 8, 8)

        label_height = 28
        img_rect = QRect(rect.left() + 6, rect.top() + 6, rect.width() - 12, rect.height() - label_height - 10)

        if is_dir:
            painter.setBrush(QBrush(QColor("#18181c")))
            painter.setPen(QPen(QColor("#27272a"), 1.0))
            painter.drawRoundedRect(img_rect, 6, 6)
            painter.setFont(QFont("Segoe UI Emoji", 42))
            painter.setPen(QColor("#f59e0b"))
            painter.drawText(img_rect, Qt.AlignCenter, "📁")
        else:
            if file_path in self.thumb_cache:
                pix = self.thumb_cache[file_path]
                scaled_pix = pix.scaled(img_rect.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                x_offset = img_rect.left() + (img_rect.width() - scaled_pix.width()) // 2
                y_offset = img_rect.top() + (img_rect.height() - scaled_pix.height()) // 2
                painter.drawPixmap(x_offset, y_offset, scaled_pix)
                if is_hidden:
                    painter.fillRect(img_rect, QColor(0, 0, 0, 160))
            else:
                painter.setBrush(QBrush(QColor("#141417")))
                painter.setPen(Qt.NoPen)
                painter.drawRoundedRect(img_rect, 6, 6)

        if not is_dir:
            # 【机内打星徽标】金色 ★ 角标，暗房模式避开右上角眼睛按钮
            rating_val = self.ratings.get(file_path, 0)
            if rating_val > 0:
                star_text = "★" * rating_val
                star_font = QFont("Segoe UI", 9, QFont.Bold)
                painter.setFont(star_font)
                fm = QFontMetrics(star_font)
                tw = fm.horizontalAdvance(star_text) + 10
                shift = 26 if (self.is_darkroom and not self.is_remove_mode) else 0
                badge_rect = QRect(rect.right() - tw - 8 - shift, rect.top() + 10, tw, 20)
                painter.setBrush(QColor(0, 0, 0, 170))
                painter.setPen(QPen(QColor("#fbbf24"), 1.2))
                painter.drawRoundedRect(badge_rect, 4, 4)
                painter.setPen(QColor("#fbbf24"))
                painter.drawText(badge_rect, Qt.AlignCenter, star_text)

            if self.is_selection_mode:
                box_rect = QRect(rect.left() + 10, rect.top() + 10, 20, 20)
                painter.setPen(QPen(QColor("#0284c7" if is_checked else "#71717a"), 1.8))
                painter.setBrush(QColor("#0284c7" if is_checked else "#202024"))
                painter.drawRoundedRect(box_rect, 4, 4)
                if is_checked:
                    painter.setPen(QPen(QColor("#ffffff"), 2.0))
                    painter.drawLine(box_rect.left() + 4, box_rect.top() + 10, box_rect.left() + 8, box_rect.top() + 15)
                    painter.drawLine(box_rect.left() + 8, box_rect.top() + 15, box_rect.left() + 15, box_rect.top() + 5)
            elif self.is_remove_mode:
                is_marked_remove = file_path in self.remove_set
                if is_marked_remove:
                    # 待移出标记：红色描边 + 淡红罩面，一眼识别
                    painter.setBrush(QColor(239, 68, 68, 55))
                    painter.setPen(QPen(QColor("#ef4444"), 2.0))
                    painter.drawRoundedRect(rect, 8, 8)
                del_rect = QRect(rect.left() + 10, rect.top() + 10, 22, 22)
                painter.setBrush(QColor("#ef4444") if is_marked_remove else QColor(0, 0, 0, 170))
                painter.setPen(QPen(QColor("#ef4444" if is_marked_remove else "#fca5a5"), 1.2))
                painter.drawRoundedRect(del_rect, 4, 4)
                painter.setPen(QPen(QColor("#ffffff"), 2.0) if is_marked_remove else QPen(QColor("#fca5a5"), 1.6))
                painter.drawLine(del_rect.left() + 5, del_rect.top() + 5, del_rect.right() - 5, del_rect.bottom() - 5)
                painter.drawLine(del_rect.right() - 5, del_rect.top() + 5, del_rect.left() + 5, del_rect.bottom() - 5)

            if self.is_darkroom and not self.is_remove_mode:
                eye_rect = QRect(rect.right() - 28, rect.top() + 10, 20, 20)
                painter.setBrush(QColor(0, 0, 0, 170))
                
                if self.cull_mode == "pick":
                    btn_col = "#38bdf8" if is_hidden else "#facc15"
                    btn_char = "＋" if is_hidden else "★"
                else:
                    btn_col = "#ef4444" if is_hidden else "#38bdf8"
                    btn_char = "✕" if is_hidden else "👁"

                painter.setPen(QPen(QColor(btn_col), 1.2))
                painter.drawRoundedRect(eye_rect, 4, 4)
                painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
                painter.setPen(QColor(btn_col))
                painter.drawText(eye_rect, Qt.AlignCenter, btn_char)

        text_rect = QRect(rect.left() + 8, rect.bottom() - label_height, rect.width() - 16, label_height)
        if is_dir:
            name_color = "#f59e0b" if is_hover else "#e4e4e7"
        else:
            name_color = "#52525b" if is_hidden else ("#f4f4f5" if is_item_selected else "#a1a1aa")
            
        painter.setPen(QColor(name_color))
        font = QFont("Segoe UI", 9)
        font.setBold(is_dir or is_item_selected or is_checked)
        painter.setFont(font)
        metrics = QFontMetrics(font)
        elided_text = metrics.elidedText(file_name, Qt.ElideMiddle, text_rect.width())
        painter.drawText(text_rect, Qt.AlignVCenter | Qt.AlignHCenter, elided_text)

        painter.restore()

class AdvancedGalleryView(QListView):
    sig_zoom_requested = Signal(int)
    sig_selection_state_changed = Signal()
    sig_card_badge_clicked = Signal(str, str)

    def update_scroll_step(self, step: int):
        self.verticalScrollBar().setSingleStep(max(15, min(250, step)))

    def __init__(self, parent=None, is_darkroom=False, cull_mode="reject"):
        super().__init__(parent)
        self.is_darkroom = is_darkroom
        self.cull_mode = cull_mode
        self.setViewMode(QListView.IconMode)
        self.setResizeMode(QListView.Adjust)
        self.setUniformItemSizes(True)
        self.setMovement(QListView.Static)
        self.setSpacing(6)
        self.setWordWrap(False)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setStyleSheet("QListView { background-color: #121214; border: none; padding: 6px; outline: none; }")

        self.delegate = AdvancedCardDelegate(self, is_darkroom=is_darkroom, cull_mode=cull_mode)
        self.setItemDelegate(self.delegate)
        self.base_card_width = 220
        self.last_clicked_row = -1
        # 读取滚轮像素步长，消除动一下就崩飞的阶梯感
        settings = QSettings("AnicropGeeks", "AnicropStudio")
        step_val = int(settings.value("mouse_scroll_step", 60))
        self.verticalScrollBar().setSingleStep(step_val)

    def set_base_card_width(self, width: int):
        self.base_card_width = max(130, min(420, width))
        self.recalculate_grid()

    def wheelEvent(self, event: QWheelEvent):
        # 主界面相册列表：Ctrl+滚轮缩放卡片大小，普通滚轮正常上下滚动相册
        if event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y()
            step = 15 if delta > 0 else -15
            self.sig_zoom_requested.emit(step)
            event.accept()
        else:
            super().wheelEvent(event)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            idx = self.indexAt(event.pos())
            is_dir = bool(idx.data(PhotoListModel.ROLE_IS_DIR)) if idx.isValid() else False

            if idx.isValid() and self.is_darkroom and self.delegate.is_remove_mode and not is_dir:
                cur_row = idx.row()
                file_path = idx.data(Qt.UserRole)
                rset = self.delegate.remove_set
                # 【快选打标】点击=切换待移出标记；Shift+点击=范围批量标记；不会立即剔除，点「完成移出」才生效
                if (event.modifiers() & Qt.ShiftModifier) and self.last_clicked_row != -1:
                    start_r = min(self.last_clicked_row, cur_row)
                    end_r = max(self.last_clicked_row, cur_row)
                    model = self.model()
                    for r in range(start_r, end_r + 1):
                        p, r_is_dir = model.get_item(r)
                        if p and not r_is_dir:
                            rset.add(p)
                else:
                    if file_path in rset:
                        rset.remove(file_path)
                    else:
                        rset.add(file_path)
                    self.last_clicked_row = cur_row
                self.sig_selection_state_changed.emit()
                self.viewport().update()
                event.accept()
                return

            if idx.isValid() and self.is_darkroom and not is_dir and not self.delegate.is_remove_mode:
                item_rect = self.visualRect(idx)
                rel_x = event.pos().x() - item_rect.left()
                rel_y = event.pos().y() - item_rect.top()
                if (rel_x >= item_rect.width() - 36) and rel_y <= 36:
                    file_path = idx.data(Qt.UserRole)
                    if file_path in self.delegate.hidden_set:
                        self.delegate.hidden_set.remove(file_path)
                    else:
                        self.delegate.hidden_set.add(file_path)
                    self.sig_selection_state_changed.emit()
                    self.viewport().update()
                    event.accept()
                    return

            if self.delegate.is_selection_mode and idx.isValid() and not is_dir:
                cur_row = idx.row()
                modifiers = event.modifiers()

                if (modifiers & Qt.ShiftModifier) and self.last_clicked_row != -1:
                    start_r = min(self.last_clicked_row, cur_row)
                    end_r = max(self.last_clicked_row, cur_row)
                    model = self.model()
                    for r in range(start_r, end_r + 1):
                        p, r_is_dir = model.get_item(r)
                        if p and not r_is_dir:
                            self.delegate.selected_set.add(p)
                    self.sig_selection_state_changed.emit()
                    self.viewport().update()
                    event.accept()
                    return

                elif modifiers & Qt.ControlModifier:
                    file_path = idx.data(Qt.UserRole)
                    if file_path in self.delegate.selected_set:
                        self.delegate.selected_set.remove(file_path)
                    else:
                        self.delegate.selected_set.add(file_path)
                    self.last_clicked_row = cur_row
                    self.sig_selection_state_changed.emit()
                    self.viewport().update()
                    event.accept()
                    return

                item_rect = self.visualRect(idx)
                rel_x = event.pos().x() - item_rect.left()
                rel_y = event.pos().y() - item_rect.top()
                if rel_x <= 36 and rel_y <= 36:
                    file_path = idx.data(Qt.UserRole)
                    if file_path in self.delegate.selected_set:
                        self.delegate.selected_set.remove(file_path)
                    else:
                        self.delegate.selected_set.add(file_path)
                    self.last_clicked_row = cur_row
                    self.sig_selection_state_changed.emit()
                    self.viewport().update()
                    event.accept()
                    return

                self.last_clicked_row = cur_row

        super().mousePressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.recalculate_grid()

    def recalculate_grid(self):
        view_width = self.viewport().width() - 12
        if view_width <= 0:
            return

        spacing = self.spacing()
        actual_width = max(120, min(view_width, self.base_card_width))
        actual_height = int(actual_width * 1.18)

        # 确保网格水平居中对齐（Qt6 标准属性，绝对不触发几何死循环崩溃）
        self.setWordWrap(False)
        
        new_size = QSize(actual_width, actual_height)
        self.setGridSize(QSize(actual_width + spacing, actual_height + spacing))
        self.delegate.set_card_size(new_size)

class FilmstripDelegate(QStyledItemDelegate):
    def __init__(self, parent=None, thumb_cache=None):
        super().__init__(parent)
        self.thumb_cache = thumb_cache if thumb_cache is not None else {}
        self.item_size = QSize(46, 50)

    def sizeHint(self, option, index):
        return self.item_size

    def paint(self, painter: QPainter, option, index):
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        rect = option.rect.adjusted(2, 3, -2, -3)
        file_path = index.data(Qt.UserRole)
        is_selected = bool(option.state & QStyle.State_Selected)

        if file_path in self.thumb_cache:
            pix = self.thumb_cache[file_path]
            scaled = pix.scaled(rect.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            sx = (scaled.width() - rect.width()) // 2
            sy = (scaled.height() - rect.height()) // 2
            painter.setClipRect(rect)
            painter.drawPixmap(rect.topLeft(), scaled, QRect(sx, sy, rect.width(), rect.height()))
        else:
            painter.fillRect(rect, QColor("#1e1e24"))

        if is_selected:
            painter.setClipping(False)
            painter.setPen(QPen(QColor("#38bdf8"), 2.2))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(rect.adjusted(-1, -1, 1, 1), 3, 3)

        painter.restore()

class FilmstripDelegate(QStyledItemDelegate):
    def __init__(self, parent=None, thumb_cache=None):
        super().__init__(parent)
        self.thumb_cache = thumb_cache if thumb_cache is not None else {}
        self.item_size = QSize(72, 78)  # 从 46x50 暴增到 72x78 (面积暴涨 2.5 倍，微表情一眼清！)

    def sizeHint(self, option, index):
        return self.item_size

    def paint(self, painter: QPainter, option, index):
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        rect = option.rect.adjusted(2, 3, -2, -3)
        file_path = index.data(Qt.UserRole)
        is_selected = bool(option.state & QStyle.State_Selected)

        if file_path in self.thumb_cache:
            pix = self.thumb_cache[file_path]
            scaled = pix.scaled(rect.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            sx = (scaled.width() - rect.width()) // 2
            sy = (scaled.height() - rect.height()) // 2
            painter.setClipRect(rect)
            painter.drawPixmap(rect.topLeft(), scaled, QRect(sx, sy, rect.width(), rect.height()))
        else:
            painter.fillRect(rect, QColor("#1e1e24"))

        if is_selected:
            painter.setClipping(False)
            painter.setPen(QPen(QColor("#38bdf8"), 2.2))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(rect.adjusted(-1, -1, 1, 1), 3, 3)

        painter.restore()

class FilmstripListView(QListView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setViewMode(QListView.IconMode)
        self.setFlow(QListView.LeftToRight)
        self.setWrapping(False)
        self.setResizeMode(QListView.Adjust)
        self.setUniformItemSizes(True)
        self.setGridSize(QSize(76, 80))  # 紧凑微间距网格
        self.setMovement(QListView.Static)
        self.setSpacing(3)
        self.setFixedHeight(86)  # 从 58px 提升到 86px 专业级高度
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setStyleSheet("QListView { background-color: rgba(10, 10, 12, 240); border-top: 1px solid #27272a; padding: 2px; outline: none; }")

    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y() or event.angleDelta().x()
        if delta != 0:
            bar = self.horizontalScrollBar()
            bar.setValue(bar.value() - int(delta * 1.5))
            event.accept()
        else:
            super().wheelEvent(event)

class MinimapNavigator(QWidget):
    sig_pan_requested = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(180, 120)
        self.pixmap = QPixmap()
        self.norm_view_rect = QRectF(0, 0, 1, 1)
        self.is_dragging = False
        self.setStyleSheet("MinimapNavigator { background-color: rgba(24, 24, 27, 210); border: 1px solid #3f3f46; border-radius: 6px; }")

    def update_state(self, pixmap: QPixmap, norm_rect: QRectF):
        self.pixmap = pixmap
        self.norm_view_rect = norm_rect
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor(18, 18, 20, 220))

        if self.pixmap.isNull():
            return

        inner_rect = self.rect().adjusted(4, 4, -4, -4)
        scaled_pix = self.pixmap.scaled(inner_rect.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        img_x = inner_rect.left() + (inner_rect.width() - scaled_pix.width()) // 2
        img_y = inner_rect.top() + (inner_rect.height() - scaled_pix.height()) // 2
        img_rect = QRect(img_x, img_y, scaled_pix.width(), scaled_pix.height())

        painter.drawPixmap(img_x, img_y, scaled_pix)

        rx = img_rect.left() + self.norm_view_rect.left() * img_rect.width()
        ry = img_rect.top() + self.norm_view_rect.top() * img_rect.height()
        rw = self.norm_view_rect.width() * img_rect.width()
        rh = self.norm_view_rect.height() * img_rect.height()

        target_box = QRectF(rx, ry, rw, rh).intersected(QRectF(img_rect))
        painter.setPen(QPen(QColor("#ef4444"), 1.8))
        painter.setBrush(QColor(239, 68, 68, 45))
        painter.drawRect(target_box)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.is_dragging = True
            self._handle_mouse_move(event.pos())

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.is_dragging:
            self._handle_mouse_move(event.pos())

    '''def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.is_dragging = False'''

    def _handle_mouse_move(self, pos: QPoint):
        inner_rect = self.rect().adjusted(4, 4, -4, -4)
        if inner_rect.width() <= 0 or inner_rect.height() <= 0:
            return
        norm_x = (pos.x() - inner_rect.left()) / inner_rect.width()
        norm_y = (pos.y() - inner_rect.top()) / inner_rect.height()
        self.sig_pan_requested.emit(max(0.0, min(1.0, norm_x)), max(0.0, min(1.0, norm_y)))

# ==============================================================================
# 置底可视化打星条 (DPP 式点星: 点 N 星打 N 星, 再点当前星级清除)
# ==============================================================================
class StarRatingBar(QWidget):
    sig_rating_clicked = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rating = 0
        self.hover_index = -1
        self.setMouseTracking(True)
        self.setFixedSize(210, 46)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("点击打星 (1~5)；再次点击当前星级可清除")

    def set_rating(self, rating: int):
        self.rating = max(0, min(5, int(rating)))
        self.update()

    def _star_index_at(self, x: int) -> int:
        base_x, size, gap = 12, 34, 4
        for i in range(5):
            if base_x + i * (size + gap) <= x < base_x + i * (size + gap) + size:
                return i
        return -1

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QColor(18, 18, 22, 230))
        painter.setPen(QPen(QColor("#2e2e36"), 1.0))
        painter.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 3, 3)

        base_x, size, gap = 12, 34, 4
        font = QFont("Segoe UI", 19, QFont.Bold)
        painter.setFont(font)
        for i in range(5):
            star_rect = QRect(base_x + i * (size + gap), 6, size, size)
            if self.hover_index >= 0:
                lit = i <= self.hover_index
            else:
                lit = i < self.rating
            color = "#fbbf24" if lit else "#52525b"
            if self.hover_index == i:
                color = "#fde047" if lit else "#71717a"
            painter.setPen(QColor(color))
            painter.drawText(star_rect, Qt.AlignCenter, "★")

    def mouseMoveEvent(self, event: QMouseEvent):
        idx = self._star_index_at(event.pos().x())
        if idx != self.hover_index:
            self.hover_index = idx
            self.update()

    def leaveEvent(self, event):
        self.hover_index = -1
        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        # 【关键】吃掉按下事件，阻断向父级检视器传播，否则单击星星会误触发照片拖拽
        if event.button() == Qt.LeftButton:
            event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            idx = self._star_index_at(event.pos().x())
            if idx >= 0:
                new_rating = 0 if (idx + 1) == self.rating else idx + 1
                self.sig_rating_clicked.emit(new_rating)

# ==============================================================================
# 单张大图检视工作台（装载【R 键视网膜级眼部定焦特写】）
# ==============================================================================
class SinglePhotoViewer(QWidget):
    sig_back_to_gallery = Signal(int)
    sig_toggle_cinema_fullscreen = Signal()
    sig_toggle_mark_requested = Signal(str)
    sig_filter_mode_toggled = Signal(bool)
    sig_photo_changed = Signal(str, int, int)
    sig_rating_requested = Signal(str, int)

    def __init__(self, parent=None, is_darkroom=False, cull_mode="reject"):
        super().__init__(parent)
        self.is_darkroom = is_darkroom
        self.cull_mode = cull_mode
        self.setStyleSheet("background-color: #09090b;")
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        self.photo_list = []
        self.current_index = -1
        self.current_pixmap = QPixmap()
        self.zoom_factor = 1.0
        self.pan_offset = QPointF(0, 0)
        self.is_dragging = False
        self.last_mouse_pos = QPoint()
        self.is_pure_fullscreen = False

        # 【核心特写状态机】
        self.is_eye_zoom_mode = False
        self.current_target_norm = (0.5, 0.5)
        self.current_target_desc = "画面中心"
        self.current_face_rect = None

        # 【对比模式状态机】左格=current_index(沿用主字段)，右格=compare_*；键盘跟随鼠标最后点击的窗格
        self.compare_mode = False
        self.compare_index = -1
        self.compare_pixmap = QPixmap()
        self.compare_zoom_factor = 1.0  # 右图专属独立缩放倍率，彻底解绑！
        self.compare_eye_zoom_mode = False  # 右图专属特写状态机
        self.compare_pan_offset = QPointF(0, 0)
        self.compare_target_norm = (0.5, 0.5)
        self.compare_target_desc = "画面中心"  # 补齐右图状态描述，彻底杜绝属性缺失！
        self.compare_face_rect = None
        self.compare_af_boxes = []
        self.active_pane = 0

        self._init_overlays()
        self.show_af_points = True
        self.current_af_boxes = []

    def _init_overlays(self):
        # 1. 顶部左侧返回键：正方形撑满 (36×36 无文字 "<")
        self.btn_back = QPushButton("<")
        self.btn_back.setFixedSize(36, 36)
        self.btn_back.setStyleSheet("""
            QPushButton {
                background-color: #18181c;
                border: none;
                border-right: 1px solid #2e2e36;
                font-family: 'Segoe UI', Consolas, sans-serif;
                font-weight: bold;
                font-size: 15px;
                color: #a1a1aa;
            }
            QPushButton:hover {
                background-color: #0284c7;
                color: #ffffff;
            }
        """)
        self.btn_back.clicked.connect(self._on_back_clicked)

        # 2. 中间纯净居中文件名 (死守中轴线)
        self.lbl_filename = QLabel("")
        self.lbl_filename.setStyleSheet("QLabel { background-color: transparent; border: none; color: #f4f4f5; font-size: 13px; font-weight: bold; font-family: 'Segoe UI', 'Consolas', sans-serif; padding: 0 4px; }")

        # 3. 双眼检测指示徽章 (与按钮等高连排)
        self.lbl_target_badge = QLabel("🎯 双眼")
        self.lbl_target_badge.setAlignment(Qt.AlignCenter)
        self.lbl_target_badge.setStyleSheet("QLabel { background-color: #142231; color: #38bdf8; border: 1px solid #0369a1; border-radius: 2px; font-size: 11px; font-weight: 600; padding: 2px 8px; min-height: 20px; max-height: 20px; }")

        self.btn_prev = QPushButton("‹", self)
        self.btn_next = QPushButton("›", self)
        arrow_style = "QPushButton { background-color: rgba(24, 24, 27, 160); color: #f4f4f5; font-size: 30px; font-weight: bold; border: none; border-radius: 25px; width: 50px; height: 50px; } QPushButton:hover { background-color: rgba(2, 132, 199, 220); }"
        self.btn_prev.setStyleSheet(arrow_style)
        self.btn_next.setStyleSheet(arrow_style)
        self.btn_prev.clicked.connect(self._on_edge_prev)
        self.btn_next.clicked.connect(self._on_edge_next)

        # 【对比模式中缝按钮】
        self.btn_cmp_left_next = QPushButton("›", self)
        self.btn_cmp_right_prev = QPushButton("‹", self)
        self.btn_cmp_left_next.setStyleSheet(arrow_style)
        self.btn_cmp_right_prev.setStyleSheet(arrow_style)
        self.btn_cmp_left_next.setVisible(False)
        self.btn_cmp_right_prev.setVisible(False)
        self.btn_cmp_left_next.clicked.connect(lambda: self._pane_step(0, 1))
        self.btn_cmp_right_prev.clicked.connect(lambda: self._pane_step(1, -1))

        self.minimap = MinimapNavigator(self)
        self.minimap.sig_pan_requested.connect(self._on_minimap_pan)

        # 4. 全屏键：移至右下角！与缩放条合体成右下控制坞
        self.btn_cinema_fullscreen = QPushButton("⛶ 全屏")
        self.btn_cinema_fullscreen.setStyleSheet("""
            QPushButton {
                background-color: #27272a;
                border: 1px solid #3f3f46;
                border-radius: 3px;
                font-weight: 600;
                font-size: 11px;
                padding: 2px 8px;
                color: #e4e4e7;
            }
            QPushButton:hover {
                background-color: #3f3f46;
                border-color: #38bdf8;
                color: #38bdf8;
            }
        """)
        self.btn_cinema_fullscreen.clicked.connect(self._toggle_fullscreen)

        # 右下角视口控制坞 (缩放 + 全屏按键一体化)
        self.zoom_box = QFrame(self)
        self.zoom_box.setStyleSheet("QFrame { background-color: rgba(18, 18, 22, 230); border: 1px solid #2e2e36; border-radius: 4px; padding: 2px 6px; }")
        zoom_layout = QHBoxLayout(self.zoom_box)
        zoom_layout.setContentsMargins(6, 2, 6, 2)
        zoom_layout.setSpacing(6)
        zoom_layout.addWidget(self.btn_cinema_fullscreen)

        self.edit_zoom = QLineEdit("100%", self.zoom_box)
        self.edit_zoom.setFixedWidth(52)
        self.edit_zoom.setAlignment(Qt.AlignCenter)
        self.edit_zoom.setStyleSheet("QLineEdit { background-color: #121214; border: 1px solid #3f3f46; border-radius: 2px; color: #38bdf8; font-weight: bold; font-size: 11px; padding: 2px; } QLineEdit:focus { border: 1px solid #38bdf8; }")
        self.edit_zoom.returnPressed.connect(self._on_zoom_text_committed)
        zoom_layout.addWidget(self.edit_zoom)

        self.btn_zoom_menu = QPushButton("▾", self.zoom_box)
        self.btn_zoom_menu.setFixedWidth(18)
        self.btn_zoom_menu.setStyleSheet("QPushButton { background-color: transparent; border: none; color: #a1a1aa; font-size: 11px; font-weight: bold; } QPushButton:hover { color: #38bdf8; }")
        self.btn_zoom_menu.clicked.connect(self._show_preset_menu)
        zoom_layout.addWidget(self.btn_zoom_menu)

        self.slider_zoom = QSlider(Qt.Horizontal, self.zoom_box)
        self.slider_zoom.setRange(75, 1000)
        self.slider_zoom.setValue(100)
        self.slider_zoom.setFixedWidth(110)
        self.slider_zoom.valueChanged.connect(self._on_slider_zoom_changed)
        zoom_layout.addWidget(self.slider_zoom)

        # 5. 单排紧凑胶囊按钮样式 (高度 22px，清爽展开)
        self._EYE_BTN_NORMAL = "QPushButton { background-color: transparent; border: 1px solid transparent; border-radius: 2px; font-weight: 600; font-size: 11px; min-height: 20px; max-height: 20px; padding: 0 8px; color: #d4d4d8; } QPushButton:hover { background-color: #27272e; color: #38bdf8; }"
        self._EYE_BTN_ACTIVE = "QPushButton { background-color: #0c4a6e; border: 1px solid #0284c7; border-radius: 2px; font-weight: 600; font-size: 11px; min-height: 20px; max-height: 20px; padding: 0 8px; color: #38bdf8; }"
        self.btn_eye_zoom = QPushButton("👁️ 特写")
        self.btn_eye_zoom.setStyleSheet(self._EYE_BTN_NORMAL)
        self.btn_eye_zoom.setToolTip(f"眼部特写 ({load_shortcuts()['eye_zoom']})")
        self.btn_eye_zoom.clicked.connect(self._toggle_eye_zoom)

        self._AF_BTN_ACTIVE = "QPushButton { background-color: #143521; border: 1px solid #16a34a; border-radius: 2px; font-weight: 600; font-size: 11px; min-height: 20px; max-height: 20px; padding: 0 8px; color: #4ade80; }"
        self._AF_BTN_INACTIVE = "QPushButton { background-color: transparent; border: 1px solid transparent; border-radius: 2px; font-weight: 600; font-size: 11px; min-height: 20px; max-height: 20px; padding: 0 8px; color: #71717a; } QPushButton:hover { background-color: #27272e; color: #d4d4d8; }"
        self.btn_af_toggle = QPushButton("🎯 对焦")
        self.btn_af_toggle.setStyleSheet(self._AF_BTN_ACTIVE)
        self.btn_af_toggle.setToolTip(f"对焦点显示 ({load_shortcuts()['af']})")
        self.btn_af_toggle.clicked.connect(self._toggle_af_points)

        self._COMPARE_BTN_NORMAL = "QPushButton { background-color: transparent; border: 1px solid transparent; border-radius: 2px; font-weight: 600; font-size: 11px; min-height: 20px; max-height: 20px; padding: 0 8px; color: #d4d4d8; } QPushButton:hover { background-color: #27272e; color: #c084fc; }"
        self._COMPARE_BTN_ACTIVE = "QPushButton { background-color: #3b185f; border: 1px solid #7c3aed; border-radius: 2px; font-weight: 600; font-size: 11px; min-height: 20px; max-height: 20px; padding: 0 8px; color: #c084fc; }"
        self.btn_compare = QPushButton("🆚 对比")
        self.btn_compare.setStyleSheet(self._COMPARE_BTN_NORMAL)
        self.btn_compare.setToolTip(f"对比模式 ({load_shortcuts()['compare']})")
        self.btn_compare.clicked.connect(self._toggle_compare)

        self.grid_mode = 0  # 0:关, 1:三分线, 2:黄金分割, 3:黄金螺旋, 4:对角三角, 5:中心十字
        self.spiral_orientation = 0  # 0:右上, 1:右下, 2:左下, 3:左上 (Shift+Q 翻转)
        self._GRID_NAMES = ["📐 构图", "📐 三分", "📐 黄金比", "📐 螺旋", "📐 九宫交叉", "📐 十字"]
        self.btn_grid_toggle = QPushButton("📐 构图")
        self.btn_grid_toggle.setStyleSheet("QPushButton { background-color: transparent; border: 1px solid transparent; border-radius: 2px; font-weight: 600; font-size: 11px; min-height: 20px; max-height: 20px; padding: 0 8px; color: #d4d4d8; } QPushButton:hover { background-color: #27272e; color: #38bdf8; }")
        self.btn_grid_toggle.setToolTip(f"诺基亚构图辅助线 ({load_shortcuts().get('grid', 'Q')})")
        self.btn_grid_toggle.clicked.connect(self._cycle_grid_mode)

        self.crop_active = False
        self.crop_ratio_mode = "4:5"  # "free", "4:5", "3:4", "1:1", "9:16", "16:9", "custom"
        self.crop_target_aspect = 4.0 / 5.0
        self.norm_crop_rect = QRectF(0.1, 0.05, 0.8, 0.9)  # 相对于原图的归一化裁切矩形 (0.0~1.0)
        self.crop_drag_mode = None  # 'move', 'top', 'bottom', 'left', 'right', 'tl', 'tr', 'bl', 'br'
        self.crop_drag_start_pos = QPointF()
        self.crop_drag_start_norm_rect = QRectF()

        self.btn_crop_toggle = QPushButton("🔲 裁切")
        self.btn_crop_toggle.setStyleSheet("QPushButton { background-color: transparent; border: 1px solid transparent; border-radius: 2px; font-weight: 600; font-size: 11px; min-height: 20px; max-height: 20px; padding: 0 8px; color: #d4d4d8; } QPushButton:hover { background-color: #27272e; color: #38bdf8; }")
        self.btn_crop_toggle.setToolTip(f"预裁剪交互模式 ({load_shortcuts().get('precrop', 'W')})")
        self.btn_crop_toggle.clicked.connect(self._toggle_crop_active)

        self.combo_crop_ratio = QComboBox()
        self.combo_crop_ratio.setVisible(False)
        self.combo_crop_ratio.setFixedHeight(22)
        self.combo_crop_ratio.setStyleSheet("""
            QComboBox {
                background-color: #141418;
                border: 1px solid #0284c7;
                border-radius: 2px;
                padding: 1px 4px;
                color: #38bdf8;
                font-weight: bold;
                font-size: 11px;
            }
            QComboBox QAbstractItemView {
                background-color: #18181c;
                border: 1px solid #3f3f46;
                selection-background-color: #0284c7;
                color: #f4f4f5;
            }
        """)
        self.combo_crop_ratio.addItem("4:5 (小红书)", "4:5")
        self.combo_crop_ratio.addItem("3:4 (人像)", "3:4")
        self.combo_crop_ratio.addItem("1:1 (正方形)", "1:1")
        self.combo_crop_ratio.addItem("9:16 (壁纸)", "9:16")
        self.combo_crop_ratio.addItem("16:9 (横屏)", "16:9")
        self.combo_crop_ratio.addItem("自由比例", "free")
        self.combo_crop_ratio.addItem("自定义比例...", "custom")
        self.combo_crop_ratio.currentIndexChanged.connect(self._on_crop_ratio_changed)
        self.combo_crop_ratio.setFocusPolicy(Qt.NoFocus)

        self.lbl_crop_live_ratio = QLabel("4:5")
        self.lbl_crop_live_ratio.setVisible(False)
        self.lbl_crop_live_ratio.setAlignment(Qt.AlignCenter)
        self.lbl_crop_live_ratio.setStyleSheet("QLabel { background-color: #142231; color: #38bdf8; border: 1px solid #0369a1; border-radius: 2px; font-size: 11px; font-weight: bold; padding: 2px 8px; min-height: 20px; max-height: 20px; font-family: Consolas, monospace; }")

        # 6. 左下角精炼浮动工具条
        self.tool_box = QFrame(self)
        self.tool_box.setStyleSheet("QFrame { background-color: rgba(18, 18, 22, 230); border: 1px solid #2e2e36; border-radius: 4px; padding: 2px 4px; }")
        tool_layout = QHBoxLayout(self.tool_box)
        tool_layout.setContentsMargins(4, 2, 4, 2)
        tool_layout.setSpacing(6)

        self.btn_filmstrip_toggle = QPushButton("🎞️ 胶卷底片", self.tool_box)
        self.btn_filmstrip_toggle.setStyleSheet("QPushButton { background-color: transparent; border: 1px solid #3f3f46; border-radius: 3px; font-weight: 600; font-size: 11px; padding: 4px 10px; color: #e4e4e7; } QPushButton:hover { background-color: #27272a; border-color: #38bdf8; color: #38bdf8; }")
        self.btn_filmstrip_toggle.clicked.connect(self._toggle_filmstrip)
        tool_layout.addWidget(self.btn_filmstrip_toggle)

        self.btn_mark_toggle = QPushButton("👁️ 状态标记", self.tool_box)
        self.btn_mark_toggle.setStyleSheet("QPushButton { background-color: transparent; border: 1px solid #3f3f46; border-radius: 3px; font-weight: 600; font-size: 11px; padding: 4px 10px; color: #38bdf8; } QPushButton:hover { background-color: #27272a; }")
        self.btn_mark_toggle.clicked.connect(self._on_mark_toggle_clicked)
        tool_layout.addWidget(self.btn_mark_toggle)

        cb_title = "显示未选照片" if self.cull_mode == "pick" else "显示废片"
        self.cb_viewer_show_unpicked = QCheckBox(cb_title, self.tool_box)
        self.cb_viewer_show_unpicked.setChecked(True)
        self.cb_viewer_show_unpicked.setStyleSheet("QCheckBox { color: #38bdf8; font-weight: bold; font-size: 11px; margin-left: 2px; }")
        self.cb_viewer_show_unpicked.toggled.connect(self._on_viewer_filter_toggled)
        self.cb_viewer_show_unpicked.setVisible(self.is_darkroom)
        tool_layout.addWidget(self.cb_viewer_show_unpicked)

        self.filmstrip_view = FilmstripListView(self)
        self.filmstrip_view.setVisible(False)
        self.filmstrip_view.clicked.connect(self._on_filmstrip_item_clicked)
        self.filmstrip_model = PhotoListModel()
        self.filmstrip_view.setModel(self.filmstrip_model)

        # 7. 中间置底固定打星条 (坚决不动)
        self.star_bar = StarRatingBar(self)
        self.star_bar.sig_rating_clicked.connect(self._on_star_bar_clicked)

        # 【全键盘驾驶舱防线】彻底剥夺所有悬浮/工具按钮的键盘焦点，杜绝方向键被抢！
        for b in (self.btn_back, self.btn_eye_zoom, self.btn_af_toggle, self.btn_compare,
                  self.btn_grid_toggle, self.btn_crop_toggle, self.btn_cinema_fullscreen, self.btn_filmstrip_toggle,
                  self.btn_mark_toggle, self.btn_prev, self.btn_next, self.btn_cmp_left_next,
                  self.btn_cmp_right_prev, self.btn_zoom_menu):
            b.setFocusPolicy(Qt.NoFocus)
        self.slider_zoom.setFocusPolicy(Qt.NoFocus)
        self.edit_zoom.setFocusPolicy(Qt.ClickFocus)
        self.cb_viewer_show_unpicked.setFocusPolicy(Qt.NoFocus)

    def _relayout_overlays(self):
        w, h = self.width(), self.height()
        # btn_back, lbl_filename, lbl_target_badge 已经托管在顶部工具栏，无需在画面内部计算坐标

        if self.compare_mode:
            half = w // 2
            self.btn_prev.move(18, (h - 50) // 2)
            self.btn_cmp_left_next.move(half - 68, (h - 50) // 2)
            self.btn_cmp_right_prev.move(half + 18, (h - 50) // 2)
            self.btn_next.move(w - 68, (h - 50) // 2)
        else:
            self.btn_prev.move(18, (h - 50) // 2)
            self.btn_next.move(w - 68, (h - 50) // 2)
        self.minimap.move(w - 196, 16)

        film_h = 86 if self.filmstrip_view.isVisible() else 0
        self.tool_box.move(16, h - 38 - film_h)
        self.zoom_box.adjustSize()
        self.zoom_box.move(w - self.zoom_box.width() - 16, h - 38 - film_h)
        self.star_bar.move((w - self.star_bar.width()) // 2, h - 48 - film_h)

        if self.filmstrip_view.isVisible():
            self.filmstrip_view.setGeometry(0, h - 86, w, 86)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._relayout_overlays()
        self._clamp_pan_offset()

    def _on_viewer_filter_toggled(self, checked: bool):
        self.sig_filter_mode_toggled.emit(checked)

    def load_context(self, index: int, photo_list: list, thumb_cache=None, is_marked: bool = False):
        _crashlog(f"[viewer] load_context idx={index} path={photo_list[index] if 0 <= index < len(photo_list) else '?'}")
        self.photo_list = list(photo_list)
        self.current_index = index

        # 换目录/换上下文时退出对比模式
        self.compare_mode = False
        self.compare_index = -1
        self.compare_pixmap = QPixmap()
        self.compare_pan_offset = QPointF(0, 0)
        self.active_pane = 0
        self.btn_compare.setText(f"🆚 对比模式 ({load_shortcuts()['compare']})")
        self.btn_compare.setStyleSheet(self._COMPARE_BTN_NORMAL)
        self.btn_cmp_left_next.setVisible(False)
        self.btn_cmp_right_prev.setVisible(False)
        # 快捷键改绑后，进入检视器时按钮文案即时刷新
        sc = load_shortcuts()
        if not self.is_eye_zoom_mode:
            self.btn_eye_zoom.setText("👁️ 特写")
        self.btn_eye_zoom.setToolTip(f"眼部特写 ({sc.get('eye_zoom', 'R')})")
        self.btn_af_toggle.setText("🎯 对焦")
        self.btn_af_toggle.setToolTip(f"对焦点显示 ({sc.get('af', 'F')})")
        self.btn_compare.setText("🆚 对比")
        self.btn_compare.setToolTip(f"对比模式 ({sc.get('compare', 'C')})")
        self.btn_grid_toggle.setToolTip(f"构图辅助线 ({sc.get('grid', 'Q')})")
        self.btn_crop_toggle.setToolTip(f"预裁剪交互模式 ({sc.get('precrop', 'W')})")

        self.filmstrip_model.set_photos(self.photo_list)
        delegate = FilmstripDelegate(self.filmstrip_view, thumb_cache=thumb_cache)
        self.filmstrip_view.setItemDelegate(delegate)

        has_multiple = len(self.photo_list) > 1
        self.btn_prev.setVisible(has_multiple and not self.is_pure_fullscreen)
        self.btn_next.setVisible(has_multiple and not self.is_pure_fullscreen)

        self._load_current_photo()
        self.update_mark_ui(is_marked)

    def update_mark_ui(self, is_marked: bool):
        if self.is_darkroom:
            if self.cull_mode == "pick":
                if is_marked:
                    self.btn_mark_toggle.setText(f"○ 未选好片 (按 {load_shortcuts()['mark']} 选定)")
                    self.btn_mark_toggle.setStyleSheet("QPushButton { background-color: transparent; border: 1px solid #52525b; border-radius: 4px; font-weight: bold; padding: 4px 10px; color: #a1a1aa; } QPushButton:hover { background-color: #27272a; }")
                else:
                    self.btn_mark_toggle.setText(f"★ 已精选好片 (按 {load_shortcuts()['mark']} 取消)")
                    self.btn_mark_toggle.setStyleSheet("QPushButton { background-color: #854d0e; border: 1px solid #facc15; border-radius: 4px; font-weight: bold; padding: 4px 10px; color: #ffffff; }")
            else:
                if is_marked:
                    self.btn_mark_toggle.setText(f"🚫 已隐藏/废片 (按 {load_shortcuts()['mark']} 恢复)")
                    self.btn_mark_toggle.setStyleSheet("QPushButton { background-color: #7f1d1d; border: 1px solid #ef4444; border-radius: 4px; font-weight: bold; padding: 4px 10px; color: #ffffff; }")
                else:
                    self.btn_mark_toggle.setText(f"👁️ 正常保留 (按 {load_shortcuts()['mark']} 隐藏)")
                    self.btn_mark_toggle.setStyleSheet("QPushButton { background-color: transparent; border: 1px solid #52525b; border-radius: 4px; font-weight: bold; padding: 4px 10px; color: #22c55e; } QPushButton:hover { background-color: #27272a; }")
        else:
            if is_marked:
                self.btn_mark_toggle.setText(f"✓ 已选择照片 (按 {load_shortcuts()['mark']} 取消)")
                self.btn_mark_toggle.setStyleSheet("QPushButton { background-color: #0369a1; border: 1px solid #38bdf8; border-radius: 4px; font-weight: bold; padding: 4px 10px; color: #ffffff; }")
            else:
                self.btn_mark_toggle.setText(f"+ 选择照片 (按 {load_shortcuts()['mark']} 勾选)")
                self.btn_mark_toggle.setStyleSheet("QPushButton { background-color: transparent; border: 1px solid #52525b; border-radius: 4px; font-weight: bold; padding: 4px 10px; color: #a1a1aa; } QPushButton:hover { background-color: #27272a; }")

    def update_rating_label(self, rating: int):
        """打星写入成功后即时刷新标题栏星级（跟随激活窗格）"""
        path = self.get_active_path()
        if not path:
            return
        name = Path(path).name
        star_str = "  " + "★" * rating if rating > 0 else ""
        tag = ""
        if self.compare_mode:
            tag = "[左] " if self.active_pane == 0 else "[右] "
        self.lbl_filename.setText(f"{tag}{name}{star_str}")
        self.star_bar.set_rating(rating)
        self._relayout_overlays()

    def _on_star_bar_clicked(self, rating: int):
        path = self.get_active_path()
        if path:
            self.sig_rating_requested.emit(path, rating)

    def _on_mark_toggle_clicked(self):
        p = self.get_active_path()
        if p:
            self.sig_toggle_mark_requested.emit(p)

    def _load_current_photo(self):
        if not (0 <= self.current_index < len(self.photo_list)):
            return

        file_path = self.photo_list[self.current_index]
        _crashlog(f"[viewer] load_photo {file_path}")
        rating_val = parse_photo_rating(file_path)
        self.star_bar.set_rating(rating_val)
        star_str = "  " + "★" * rating_val if rating_val > 0 else ""
        self.lbl_filename.setText(f"{Path(file_path).name}{star_str}")

        img = load_photo_qimage(file_path)
        _crashlog(f"[viewer] 解码完成 {img.width()}x{img.height()} null={img.isNull()}")
        self.current_pixmap = QPixmap.fromImage(img) if not img.isNull() else QPixmap()

        if is_raw_file(file_path):
            # RAW：方向/佳能 AF 框从容器 EXIF 照解；人脸雷达直接跑在内嵌预览上（与显示严格同源）
            orientation = extract_raw_orientation(file_path)
            self.current_af_boxes = extract_canon_af_points(file_path, orientation)
            _crashlog(f"[viewer] 人脸雷达开始(RAW) {file_path}")
            loc_res = FocusInspector.locate_target_from_qimage(img, self.current_af_boxes)
            _crashlog("[viewer] 人脸雷达完成")
            self.current_target_norm = loc_res["target_norm"]
            self.current_target_desc = loc_res["target_desc"] if loc_res["target_desc"] != "画面中心" else "RAW 内嵌预览 (居中)"
            self.current_face_rect = loc_res["debug_face_rect"]
        else:
            _, orientation = extract_exif_thumb_and_orientation(file_path)
            self.current_af_boxes = extract_canon_af_points(file_path, orientation)

            # 【核心：极速雷达瞄准双眼】
            _crashlog(f"[viewer] 人脸雷达开始 {file_path}")
            loc_res = FocusInspector.locate_target(file_path, self.current_af_boxes, orientation)
            _crashlog("[viewer] 人脸雷达完成")
            self.current_target_norm = loc_res["target_norm"]
            self.current_target_desc = loc_res["target_desc"]
            self.current_face_rect = loc_res["debug_face_rect"]

        if self.filmstrip_model.rowCount() > 0:
            idx = self.filmstrip_model.index(self.current_index, 0)
            self.filmstrip_view.setCurrentIndex(idx)
            if self.filmstrip_view.isVisible():
                self.filmstrip_view.scrollTo(idx, QAbstractItemView.PositionAtCenter)

        # 【切图视口继承】：如果正处于特写模式，下一张照片瞬间自动对准该图双眼！
        if self.is_eye_zoom_mode:
            self._apply_eye_zoom()
        elif self.compare_mode:
            # 对比模式切图：保持当前倍率，双格各自夹紧并刷新激活窗格 UI
            self._clamp_all_pans()
            self._sync_zoom_ui()
            self._refresh_active_pane_ui()
            self.update()
        else:
            self.zoom_factor = 1.0
            self.pan_offset = QPointF(0, 0)
            self._clamp_pan_offset()
            self._sync_zoom_ui()
            self._sync_minimap()
            self.lbl_target_badge.setText(f"🎯 {self.current_target_desc}")
            self.lbl_target_badge.setStyleSheet("QLabel { background-color: #142231; color: #38bdf8; border: 1px solid #0369a1; border-radius: 2px; padding: 1px 6px; font-size: 11px; font-weight: 600; }")
            self._relayout_overlays()
            self.update()

        self.sig_photo_changed.emit(file_path, self.current_index, len(self.photo_list))

    def _apply_eye_zoom(self):
        """视网膜自适应特写：精准绑定当前激活窗口，单格特写，绝不动另一格！"""
        if self.compare_mode:
            pane = self.active_pane
            pr = self._pane_rect(pane)
            base_sz = self._get_pane_base_scaled_size(pane)
            face = self.current_face_rect if pane == 0 else self.compare_face_rect
            if face is not None and face.width() > 1e-4 and base_sz.width() > 0:
                target_face_px = pr.width() * 0.58
                current_face_px = face.width() * base_sz.width()
                adaptive_zoom = target_face_px / max(current_face_px, 1.0)
                pane_zoom = max(1.5, min(10.0, adaptive_zoom))
            else:
                pane_zoom = 2.5

            self._set_pane_zoom(pane, pane_zoom)
            self._set_pane_eye_zoom(pane, True)

            bs = self._get_pane_base_scaled_size(pane)
            nx, ny = self.current_target_norm if pane == 0 else self.compare_target_norm
            draw_w = bs.width() * pane_zoom
            draw_h = bs.height() * pane_zoom
            self._set_pane_pan(pane, QPointF((0.5 - nx) * draw_w, (0.5 - ny) * draw_h))
            self._clamp_pane_pan(pane)
            self._sync_zoom_ui()
            self._sync_eye_zoom_ui()
            self.update()
        else:
            self.is_eye_zoom_mode = True
            base_sz = self._get_base_scaled_size()
            if self.current_face_rect is not None and self.current_face_rect.width() > 1e-4 and base_sz.width() > 0:
                target_face_px = self.width() * 0.58
                current_face_px = self.current_face_rect.width() * base_sz.width()
                adaptive_zoom = target_face_px / max(current_face_px, 1.0)
                self.zoom_factor = max(1.5, min(10.0, adaptive_zoom))
            else:
                self.zoom_factor = 2.5

            draw_w = base_sz.width() * self.zoom_factor
            draw_h = base_sz.height() * self.zoom_factor
            nx, ny = self.current_target_norm
            self.pan_offset = QPointF((0.5 - nx) * draw_w, (0.5 - ny) * draw_h)
            self._clamp_pan_offset()
            self._sync_zoom_ui()
            self._sync_minimap()
            self._sync_eye_zoom_ui()
            self.update()

    def _reset_to_fit(self):
        """退出特写，还原适配全景（对比模式下仅还原当前激活窗口）"""
        if self.compare_mode:
            pane = self.active_pane
            self._set_pane_zoom(pane, 1.0)
            self._set_pane_eye_zoom(pane, False)
            self._set_pane_pan(pane, QPointF(0, 0))
            self._clamp_pane_pan(pane)
            self._sync_zoom_ui()
            self._sync_eye_zoom_ui()
            self.update()
        else:
            self.is_eye_zoom_mode = False
            self.zoom_factor = 1.0
            self.pan_offset = QPointF(0, 0)
            self._clamp_pan_offset()
            self._sync_zoom_ui()
            self._sync_minimap()
            self._sync_eye_zoom_ui()
            self.update()

    def _toggle_eye_zoom(self):
        """特写开关：依据当前激活窗格的倍率独立切换特写/还原"""
        pane = self.active_pane if self.compare_mode else 0
        cur_z = self._pane_zoom(pane)
        cur_eye = self._is_pane_eye_zoom(pane)
        # 核心逻辑：当前窗口处于特写且已放大，按下必定还原！否则进入特写！
        if cur_eye and cur_z > 1.2:
            self._reset_to_fit()
        else:
            self._apply_eye_zoom()

    def get_active_path(self) -> str:
        """当前激活窗格（鼠标最后点击的一侧）的照片路径；普通模式即主照片"""
        idx = self.current_index
        if self.compare_mode and self.active_pane == 1:
            idx = self.compare_index
        if 0 <= idx < len(self.photo_list):
            return self.photo_list[idx]
        return ""

    def _pane_rect(self, pane: int) -> QRectF:
        """窗格几何：普通模式=整个视口；对比模式 0=左半 1=右半"""
        if not self.compare_mode:
            return QRectF(self.rect())
        half = self.width() // 2
        if pane == 0:
            return QRectF(0, 0, half, self.height())
        return QRectF(half, 0, self.width() - half, self.height())

    def _pane_pixmap(self, pane: int) -> QPixmap:
        return self.compare_pixmap if (self.compare_mode and pane == 1) else self.current_pixmap

    def _pane_index(self, pane: int) -> int:
        return self.compare_index if (self.compare_mode and pane == 1) else self.current_index

    def _set_pane_pan(self, pane: int, pan: QPointF):
        """设置指定窗格的独立平移偏移量"""
        if self.compare_mode and pane == 1:
            self.compare_pan_offset = QPointF(pan)
        else:
            self.pan_offset = QPointF(pan)

    def _pane_zoom(self, pane: int) -> float:
        """获取指定窗格的独立缩放倍率"""
        return self.compare_zoom_factor if (self.compare_mode and pane == 1) else self.zoom_factor

    def _set_pane_zoom(self, pane: int, zoom: float):
        """设置指定窗格的独立缩放倍率"""
        if self.compare_mode and pane == 1:
            self.compare_zoom_factor = zoom
        else:
            self.zoom_factor = zoom

    def _is_pane_eye_zoom(self, pane: int) -> bool:
        """获取指定窗格是否处于眼部特写状态"""
        return self.compare_eye_zoom_mode if (self.compare_mode and pane == 1) else self.is_eye_zoom_mode

    def _set_pane_eye_zoom(self, pane: int, is_zoom: bool):
        """设置指定窗格的眼部特写状态"""
        if self.compare_mode and pane == 1:
            self.compare_eye_zoom_mode = is_zoom
        else:
            self.is_eye_zoom_mode = is_zoom

    def _sync_eye_zoom_ui(self):
        """根据当前激活窗格的状态，动态刷新右上角按钮与徽章"""
        pane = self.active_pane if self.compare_mode else 0
        is_zoom = self._is_pane_eye_zoom(pane)
        z = self._pane_zoom(pane)
        desc = getattr(self, "current_target_desc", "双眼") if pane == 0 else getattr(self, "compare_target_desc", "双眼")

        if is_zoom and z > 1.2:
            self.btn_eye_zoom.setText("🔍 还原")
            self.btn_eye_zoom.setStyleSheet(self._EYE_BTN_ACTIVE)
            self.lbl_target_badge.setText(f"🎯 {int(z * 100)}% 锁定")
            self.lbl_target_badge.setStyleSheet("QLabel { background-color: #0369a1; color: #ffffff; border: 1px solid #38bdf8; border-radius: 2px; padding: 1px 6px; font-size: 11px; font-weight: 600; }")
        else:
            self.btn_eye_zoom.setText("👁️ 特写")
            self.btn_eye_zoom.setStyleSheet(self._EYE_BTN_NORMAL)
            self.lbl_target_badge.setText(f"🎯 {desc}")
            self.lbl_target_badge.setStyleSheet("QLabel { background-color: #142231; color: #38bdf8; border: 1px solid #0369a1; border-radius: 2px; padding: 1px 6px; font-size: 11px; font-weight: 600; }")

    def _get_pane_base_scaled_size(self, pane: int) -> QSizeF:
        """指定窗格内照片 letterbox 适配后的基准尺寸"""
        pm = self._pane_pixmap(pane)
        if pm.isNull():
            return QSizeF(0, 0)
        pr = self._pane_rect(pane)
        if pr.width() <= 0 or pr.height() <= 0:
            return QSizeF(0, 0)
        scale = min(pr.width() / pm.width(), pr.height() / pm.height())
        return QSizeF(pm.width() * scale, pm.height() * scale)

    def _clamp_pane_pan(self, pane: int):
        """夹紧平移量：保证平移自由度，超界限制边缘，杜绝强制锁死(0.0)导致无法平移！"""
        if self._pane_pixmap(pane).isNull():
            return
        base_sz = self._get_pane_base_scaled_size(pane)
        z = self._pane_zoom(pane)
        draw_w = base_sz.width() * z
        draw_h = base_sz.height() * z
        pr = self._pane_rect(pane)
        pan = QPointF(self.compare_pan_offset if (self.compare_mode and pane == 1) else self.pan_offset)

        # 动态裕量：不论是否超界，都留出充足的漫游空间，消除假死
        limit_x = max(abs(draw_w - pr.width()) / 2.0, pr.width() * 0.4) + 60.0
        limit_y = max(abs(draw_h - pr.height()) / 2.0, pr.height() * 0.4) + 60.0

        pan.setX(max(-limit_x, min(limit_x, pan.x())))
        pan.setY(max(-limit_y, min(limit_y, pan.y())))

        self._set_pane_pan(pane, pan)

    def _clamp_all_pans(self):
        self._clamp_pane_pan(0)
        if self.compare_mode:
            self._clamp_pane_pan(1)

    def _route_pan_delta(self, delta: QPointF):
        """位移路由：对比模式下严格物理隔离，只移动当前激活/交互的窗格，杜绝任何联动！"""
        if self.compare_mode:
            pan = QPointF(self.compare_pan_offset if self.active_pane == 1 else self.pan_offset)
            self._set_pane_pan(self.active_pane, pan + delta)
            self._clamp_pane_pan(self.active_pane)
        else:
            self.pan_offset += delta
            self._clamp_pan_offset()
        self._sync_minimap()
        self.update()

    def _refresh_active_pane_ui(self):
        """对比模式下：标题栏/打星条/标记按钮同步到激活窗格的照片"""
        if not self.compare_mode:
            return
        path = self.get_active_path()
        if not path:
            return
        rating_val = parse_photo_rating(path)
        self.star_bar.set_rating(rating_val)
        star_str = "  " + "★" * rating_val if rating_val > 0 else ""
        tag = "[左] " if self.active_pane == 0 else "[右] "
        self.lbl_filename.setText(f"{tag}{Path(path).name}{star_str}")
        self._relayout_overlays()

    def _toggle_compare(self):
        """对比模式：左右分屏，右格自动选下一张；键盘/打星/标记跟随鼠标最后点击的窗格"""
        if not self.photo_list:
            return
        if not self.compare_mode:
            if len(self.photo_list) < 2:
                return
            self.compare_mode = True
            self.active_pane = 0
            self.compare_index = (self.current_index + 1) % len(self.photo_list)
            self._load_compare_photo(self.compare_index)
            self.compare_pan_offset = QPointF(self.pan_offset)
            self.btn_compare.setText("🆚 退出对比")
            self.btn_compare.setStyleSheet(self._COMPARE_BTN_ACTIVE)
            self.minimap.setVisible(False)
            show_arrows = not self.is_pure_fullscreen
            self.btn_cmp_left_next.setVisible(show_arrows)
            self.btn_cmp_right_prev.setVisible(show_arrows)
            if self.is_eye_zoom_mode:
                self._apply_eye_zoom()
            else:
                self._clamp_all_pans()
            self._refresh_active_pane_ui()
        else:
            self.compare_mode = False
            self.btn_compare.setText("🆚 对比")
            self.btn_compare.setStyleSheet(self._COMPARE_BTN_NORMAL)
            self.minimap.setVisible(not self.is_pure_fullscreen or self.zoom_factor > 1.05)
            self.btn_cmp_left_next.setVisible(False)
            self.btn_cmp_right_prev.setVisible(False)
            # 退出对比后标题栏还原为主照片
            if 0 <= self.current_index < len(self.photo_list):
                path = self.photo_list[self.current_index]
                rating_val = parse_photo_rating(path)
                self.star_bar.set_rating(rating_val)
                star_str = "  " + "★" * rating_val if rating_val > 0 else ""
                self.lbl_filename.setText(f"{Path(path).name}{star_str}")
                self.sig_toggle_mark_requested.emit("__CHECK_STATE__")
            self._clamp_pan_offset()
        self._sync_zoom_ui()
        self._relayout_overlays()
        self.update()

    def _load_compare_photo(self, index: int):
        """加载右格对比照片：解码、佳能 AF 框、双眼锚点（与左格同一空间体系）"""
        if not (0 <= index < len(self.photo_list)):
            self.compare_pixmap = QPixmap()
            return
        self.compare_index = index
        file_path = self.photo_list[index]

        img = load_photo_qimage(file_path)
        self.compare_pixmap = QPixmap.fromImage(img) if not img.isNull() else QPixmap()

        if is_raw_file(file_path):
            orientation = extract_raw_orientation(file_path)
            self.compare_af_boxes = extract_canon_af_points(file_path, orientation)
            loc_res = FocusInspector.locate_target_from_qimage(img, self.compare_af_boxes)
            self.compare_target_norm = loc_res["target_norm"]
            self.compare_target_desc = loc_res.get("target_desc", "画面中心")
            self.compare_face_rect = loc_res["debug_face_rect"]
        else:
            _, orientation = extract_exif_thumb_and_orientation(file_path)
            self.compare_af_boxes = extract_canon_af_points(file_path, orientation)

            loc_res = FocusInspector.locate_target(file_path, self.compare_af_boxes, orientation)
            self.compare_target_norm = loc_res["target_norm"]
            self.compare_target_desc = loc_res.get("target_desc", "画面中心")
            self.compare_face_rect = loc_res["debug_face_rect"]

        if not self.is_eye_zoom_mode:
            self.compare_pan_offset = QPointF(0, 0)

    def _on_edge_prev(self):
        """左边缘 ‹：对比模式=左格上一张；普通模式=主照片上一张"""
        if self.compare_mode:
            self._pane_step(0, -1)
        else:
            self.prev_image()

    def _on_edge_next(self):
        """右边缘 ›：对比模式=右格下一张；普通模式=主照片下一张"""
        if self.compare_mode:
            self._pane_step(1, 1)
        else:
            self.next_image()

    def _pane_step(self, pane: int, delta: int):
        """箭头按钮翻页：先把激活窗格切到按钮所属的一侧（键盘焦点跟随），再翻页"""
        if self.active_pane != pane:
            self.active_pane = pane
            self._refresh_active_pane_ui()
        self._compare_step(delta, pane=pane)

    def _compare_step(self, delta: int, pane: int = None):
        """对比模式翻页：翻动指定窗格（默认激活窗格）的照片，自动跳过另一窗格正在看的照片"""
        n = len(self.photo_list)
        if n < 2:
            return
        if pane is None:
            pane = self.active_pane
        other = self.compare_index if pane == 0 else self.current_index
        idx = (self.current_index if pane == 0 else self.compare_index) + delta
        while 0 <= idx < n and idx == other:
            idx += delta
        if not (0 <= idx < n):
            return

        if pane == 0:
            self.current_index = idx
            self._load_current_photo()
        else:
            self._load_compare_photo(idx)
            if self.is_eye_zoom_mode:
                self._apply_eye_zoom()
            else:
                self._clamp_all_pans()
            self._refresh_active_pane_ui()
            self.update()
        self.sig_toggle_mark_requested.emit("__CHECK_STATE__")

    def _zoom_compare_pane(self, cursor_pos: QPointF, multiplier: float):
        """对比模式缩放：以光标所在窗格为锚；非特写时双格共享位移"""
        old_zoom = self.zoom_factor
        new_zoom = max(0.75, min(10.0, old_zoom * multiplier))
        if abs(new_zoom - old_zoom) < 1e-4:
            return

        pane = 1 if self._pane_rect(1).contains(cursor_pos) else 0
        pr = self._pane_rect(pane)
        base_sz = self._get_pane_base_scaled_size(pane)
        if base_sz.width() <= 0:
            return
        pan = self.compare_pan_offset if pane == 1 else self.pan_offset

        old_left = pr.x() + (pr.width() - base_sz.width() * old_zoom) / 2.0 + pan.x()
        old_top = pr.y() + (pr.height() - base_sz.height() * old_zoom) / 2.0 + pan.y()
        rel_x = (cursor_pos.x() - old_left) / old_zoom
        rel_y = (cursor_pos.y() - old_top) / old_zoom

        self.zoom_factor = new_zoom

        new_left = cursor_pos.x() - rel_x * new_zoom
        new_top = cursor_pos.y() - rel_y * new_zoom
        new_pan = QPointF(
            new_left - pr.x() - (pr.width() - base_sz.width() * new_zoom) / 2.0,
            new_top - pr.y() - (pr.height() - base_sz.height() * new_zoom) / 2.0
        )
        self._set_pane_pan(pane, new_pan)
        self._clamp_pane_pan(pane)
        self._sync_zoom_ui()
        self.update()

    def prev_image(self):
        if self.compare_mode:
            self._compare_step(-1)
            return
        if self.current_index > 0:
            self.current_index -= 1
            self._load_current_photo()
            self.sig_toggle_mark_requested.emit("__CHECK_STATE__")

    def next_image(self):
        if self.compare_mode:
            self._compare_step(1)
            return
        if self.current_index < len(self.photo_list) - 1:
            self.current_index += 1
            self._load_current_photo()
            self.sig_toggle_mark_requested.emit("__CHECK_STATE__")

    def _on_back_clicked(self):
        if self.compare_mode:
            self._toggle_compare()
            return
        self.sig_back_to_gallery.emit(self.current_index)

    def _toggle_filmstrip(self):
        is_vis = not self.filmstrip_view.isVisible()
        self.filmstrip_view.setVisible(is_vis)
        self.btn_filmstrip_toggle.setText("🎞️ 隐藏胶卷" if is_vis else "🎞️ 胶卷底片")
        self._relayout_overlays()

        if is_vis:
            self.filmstrip_view.doItemsLayout()
            if self.filmstrip_model.rowCount() > 0:
                idx = self.filmstrip_model.index(self.current_index, 0)
                self.filmstrip_view.setCurrentIndex(idx)
                self.filmstrip_view.scrollTo(idx, QAbstractItemView.PositionAtCenter)

    def _toggle_fullscreen(self):
        self.sig_toggle_cinema_fullscreen.emit()

    def set_pure_fullscreen_mode(self, is_pure: bool):
        self.is_pure_fullscreen = is_pure
        for w in [self.btn_back, self.lbl_filename, self.lbl_target_badge, self.zoom_box, self.tool_box, self.filmstrip_view, self.star_bar]:
            w.setVisible(not is_pure)

        has_multiple = len(self.photo_list) > 1
        self.btn_prev.setVisible(has_multiple and not is_pure)
        self.btn_next.setVisible(has_multiple and not is_pure)
        self.btn_cmp_left_next.setVisible(has_multiple and not is_pure and self.compare_mode)
        self.btn_cmp_right_prev.setVisible(has_multiple and not is_pure and self.compare_mode)

        if is_pure:
            self.minimap.setVisible(self.zoom_factor > 1.05 and not self.compare_mode)
        else:
            self.minimap.setVisible(not self.compare_mode)
        self.update()

    def _on_filmstrip_item_clicked(self, index: QModelIndex):
        if not index.isValid():
            return
        if self.compare_mode:
            row = index.row()
            other = self.compare_index if self.active_pane == 0 else self.current_index
            if row == other:
                return  # 不能和另一窗格看同一张
            if self.active_pane == 0:
                if row != self.current_index:
                    self.current_index = row
                    self._load_current_photo()
            else:
                if row != self.compare_index:
                    self._load_compare_photo(row)
                    if self.is_eye_zoom_mode:
                        self._apply_eye_zoom()
                    else:
                        self._clamp_all_pans()
                    self._refresh_active_pane_ui()
                    self.update()
            self.sig_toggle_mark_requested.emit("__CHECK_STATE__")
            return
        if index.row() != self.current_index:
            self.current_index = index.row()
            self._load_current_photo()
            self.sig_toggle_mark_requested.emit("__CHECK_STATE__")

    def _show_preset_menu(self):
        menu = QMenu(self)
        menu.setStyleSheet("QMenu { background-color: #18181b; border: 1px solid #3f3f46; color: #e4e4e7; padding: 4px; } QMenu::item:selected { background-color: #0284c7; color: #ffffff; }")
        presets = [
            ("🔲 适配窗口 (居中)", 1.0),
            ("👁️ 双眼特写 (250%)", 2.5),
            ("🔍 100% 原始像素", 1.0),
            ("🔎 200%", 2.0),
            ("🔎 400%", 4.0),
        ]
        for text, factor in presets:
            action = QAction(text, self)
            if factor == 2.5:
                action.triggered.connect(lambda checked=False: self._apply_eye_zoom())
            else:
                action.triggered.connect(lambda checked=False, f=factor: self._apply_target_zoom(f))
            menu.addAction(action)
        menu.exec(self.btn_zoom_menu.mapToGlobal(QPoint(0, self.btn_zoom_menu.height())))

    def _apply_target_zoom(self, target_factor: float):
        self.zoom_factor = max(0.75, min(10.0, target_factor))
        self.pan_offset = QPointF(0, 0)
        self._clamp_pan_offset()
        self._sync_zoom_ui()
        self._sync_minimap()
        self.update()

    def _on_zoom_text_committed(self):
        text = self.edit_zoom.text().replace("%", "").strip()
        try:
            val = float(text)
            val = max(75.0, min(1000.0, val))
            self._apply_target_zoom(val / 100.0)
        except ValueError:
            self._sync_zoom_ui()

    def _sync_zoom_ui(self):
        z = self._pane_zoom(self.active_pane) if self.compare_mode else self.zoom_factor
        pct = int(z * 100)
        self.edit_zoom.setText(f"{pct}%")
        self.slider_zoom.blockSignals(True)
        self.slider_zoom.setValue(pct)
        self.slider_zoom.blockSignals(False)

    def _clamp_pan_offset(self):
        self._clamp_pane_pan(0)

    def _get_base_scaled_size(self) -> QSizeF:
        if self.current_pixmap.isNull():
            return QSizeF(0, 0)
        win_w, win_h = self.width(), self.height()
        pix_w, pix_h = self.current_pixmap.width(), self.current_pixmap.height()
        scale = min(win_w / pix_w, win_h / pix_h)
        return QSizeF(pix_w * scale, pix_h * scale)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor("#050507"))

        if self.compare_mode:
            self._paint_compare(painter)
            return

        if self.current_pixmap.isNull():
            painter.setPen(QColor("#71717a"))
            painter.drawText(self.rect(), Qt.AlignCenter, "无法解析原图数据")
            return

        base_sz = self._get_base_scaled_size()
        draw_w = base_sz.width() * self.zoom_factor
        draw_h = base_sz.height() * self.zoom_factor

        win_w, win_h = self.width(), self.height()
        draw_x = (win_w - draw_w) / 2.0 + self.pan_offset.x()
        draw_y = (win_h - draw_h) / 2.0 + self.pan_offset.y()

        painter.drawPixmap(QRectF(draw_x, draw_y, draw_w, draw_h).toRect(), self.current_pixmap)

        # 诺基亚构图辅助线 (若开启裁切模式，自动隐藏全局线，避免打架)
        if self.grid_mode > 0 and not self.crop_active:
            painter.device().spiral_orientation = getattr(self, "spiral_orientation", 0)
            self._draw_lumia_grid(painter, QRectF(draw_x, draw_y, draw_w, draw_h), self.grid_mode)

        # 2. 交互式预裁剪 50% 阴影遮罩渲染
        if self.crop_active:
            self._draw_interactive_crop(painter, QRectF(draw_x, draw_y, draw_w, draw_h))

        if self.show_af_points and self.current_af_boxes:
            painter.setRenderHint(QPainter.Antialiasing)
            for box_data in self.current_af_boxes:
                norm_r = box_data["rect"]
                bx = draw_x + norm_r.x() * draw_w
                by = draw_y + norm_r.y() * draw_h
                bw = norm_r.width() * draw_w
                bh = norm_r.height() * draw_h
                target_rect = QRectF(bx, by, bw, bh)

                is_in_focus = box_data["in_focus"]
                pen_color = QColor("#22c55e") if is_in_focus else QColor("#eab308")

                painter.setPen(QPen(pen_color, 1.5))
                painter.setBrush(QColor(34, 197, 94, 28) if is_in_focus else Qt.NoBrush)
                painter.drawRect(target_rect)

                corner_len = min(14.0, bw * 0.25, bh * 0.25)
                painter.setPen(QPen(pen_color, 2.5))
                painter.drawLine(QPointF(bx, by), QPointF(bx + corner_len, by))
                painter.drawLine(QPointF(bx, by), QPointF(bx, by + corner_len))
                painter.drawLine(QPointF(bx + bw, by), QPointF(bx + bw - corner_len, by))
                painter.drawLine(QPointF(bx + bw, by), QPointF(bx + bw, by + corner_len))
                painter.drawLine(QPointF(bx, by + bh), QPointF(bx + corner_len, by + bh))
                painter.drawLine(QPointF(bx, by + bh), QPointF(bx, by + bh - corner_len))
                painter.drawLine(QPointF(bx + bw, by + bh), QPointF(bx + bw - corner_len, by + bh))
                painter.drawLine(QPointF(bx + bw, by + bh), QPointF(bx + bw, by + bh - corner_len))

        # 绘制人脸雷达目标指示框（遵循设置项开关，且彻底移除文字干扰）
        settings = QSettings("AnicropGeeks", "AnicropStudio")
        show_face_rect = settings.value("show_face_radar_rect", True, type=bool)
        if show_face_rect and self.show_af_points and self.current_face_rect and not self.is_eye_zoom_mode:
            fr = self.current_face_rect
            fx = draw_x + fr.x() * draw_w
            fy = draw_y + fr.y() * draw_h
            fw = fr.width() * draw_w
            fh = fr.height() * draw_h
            painter.setPen(QPen(QColor("#38bdf8"), 1.2, Qt.DashLine))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(QRectF(fx, fy, fw, fh))

    def _paint_compare(self, painter: QPainter):
        """对比模式绘制：左右双窗格各自 letterbox 绘制 + AF 框 + 激活高亮边 + 窗格顶标 + 中缝"""
        for pane in (0, 1):
            pr = self._pane_rect(pane)
            pm = self._pane_pixmap(pane)
            painter.fillRect(pr, QColor("#050507"))

            if pm.isNull():
                painter.setPen(QColor("#71717a"))
                painter.drawText(pr, Qt.AlignCenter, "无法解析原图数据")
            else:
                base_sz = self._get_pane_base_scaled_size(pane)
                z = self._pane_zoom(pane)
                draw_w = base_sz.width() * z
                draw_h = base_sz.height() * z
                pan = self.pan_offset if pane == 0 else self.compare_pan_offset
                draw_x = pr.x() + (pr.width() - draw_w) / 2.0 + pan.x()
                draw_y = pr.y() + (pr.height() - draw_h) / 2.0 + pan.y()

                painter.save()
                painter.setClipRect(pr)
                painter.drawPixmap(QRectF(draw_x, draw_y, draw_w, draw_h).toRect(), pm)

                if self.grid_mode > 0 and not self.crop_active:
                    painter.device().spiral_orientation = getattr(self, "spiral_orientation", 0)
                    self._draw_lumia_grid(painter, QRectF(draw_x, draw_y, draw_w, draw_h), self.grid_mode)

                af_boxes = self.current_af_boxes if pane == 0 else self.compare_af_boxes
                if self.show_af_points and af_boxes:
                    painter.setRenderHint(QPainter.Antialiasing)
                    for box_data in af_boxes:
                        norm_r = box_data["rect"]
                        bx = draw_x + norm_r.x() * draw_w
                        by = draw_y + norm_r.y() * draw_h
                        bw = norm_r.width() * draw_w
                        bh = norm_r.height() * draw_h
                        target_rect = QRectF(bx, by, bw, bh)

                        is_in_focus = box_data["in_focus"]
                        pen_color = QColor("#22c55e") if is_in_focus else QColor("#eab308")

                        painter.setPen(QPen(pen_color, 1.5))
                        painter.setBrush(QColor(34, 197, 94, 28) if is_in_focus else Qt.NoBrush)
                        painter.drawRect(target_rect)

                        corner_len = min(14.0, bw * 0.25, bh * 0.25)
                        painter.setPen(QPen(pen_color, 2.5))
                        painter.drawLine(QPointF(bx, by), QPointF(bx + corner_len, by))
                        painter.drawLine(QPointF(bx, by), QPointF(bx, by + corner_len))
                        painter.drawLine(QPointF(bx + bw, by), QPointF(bx + bw - corner_len, by))
                        painter.drawLine(QPointF(bx + bw, by), QPointF(bx + bw, by + corner_len))
                        painter.drawLine(QPointF(bx, by + bh), QPointF(bx + corner_len, by + bh))
                        painter.drawLine(QPointF(bx, by + bh), QPointF(bx, by + bh - corner_len))
                        painter.drawLine(QPointF(bx + bw, by + bh), QPointF(bx + bw - corner_len, by + bh))
                        painter.drawLine(QPointF(bx + bw, by + bh), QPointF(bx + bw, by + bh - corner_len))

                face_rect = self.current_face_rect if pane == 0 else self.compare_face_rect
                settings = QSettings("AnicropGeeks", "AnicropStudio")
                show_face_rect = settings.value("show_face_radar_rect", True, type=bool)
                if show_face_rect and self.show_af_points and face_rect and not self.is_eye_zoom_mode:
                    fx = draw_x + face_rect.x() * draw_w
                    fy = draw_y + face_rect.y() * draw_h
                    fw = face_rect.width() * draw_w
                    fh = face_rect.height() * draw_h
                    painter.setPen(QPen(QColor("#38bdf8"), 1.2, Qt.DashLine))
                    painter.setBrush(Qt.NoBrush)
                    painter.drawRect(QRectF(fx, fy, fw, fh))

                painter.restore()

            # 激活窗格高亮边框
            if pane == self.active_pane:
                painter.setPen(QPen(QColor("#38bdf8"), 2))
                painter.setBrush(Qt.NoBrush)
                painter.drawRect(pr.adjusted(1, 1, -2, -2))

            # 窗格顶标
            idx = self._pane_index(pane)
            if 0 <= idx < len(self.photo_list):
                tag = "[左] " if pane == 0 else "[右] "
                painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
                painter.setPen(QColor("#38bdf8") if pane == self.active_pane else QColor("#a1a1aa"))
                painter.drawText(pr.adjusted(12, 40, -12, 0), Qt.AlignTop | Qt.AlignLeft, tag + Path(self.photo_list[idx]).name)

        # 中缝分割线
        cx = self.width() // 2
        painter.setPen(QPen(QColor("#3f3f46"), 1))
        painter.drawLine(cx, 0, cx, self.height())

    def wheelEvent(self, event: QWheelEvent):
        # 1. 对比模式：自动将当前操作窗口对准鼠标悬停的图
        if self.compare_mode:
            hover_pane = 1 if self._pane_rect(1).contains(event.position()) else 0
            if hover_pane != self.active_pane:
                self.active_pane = hover_pane
                self._refresh_active_pane_ui()
                self._sync_zoom_ui()
                self._sync_eye_zoom_ui()

        # 2. 按住 Ctrl + 滚轮 = 严格以光标为锚点顺滑缩放 (零漂移、指哪打哪！)
        if event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y()
            if delta != 0:
                zoom_multiplier = 1.15 if delta > 0 else (1.0 / 1.15)
                if self.compare_mode:
                    self._zoom_compare_at_cursor(self.active_pane, event.position(), zoom_multiplier)
                else:
                    self._zoom_at_cursor(event.position(), zoom_multiplier)
            event.accept()
        # 3. 按住 Shift + 滚轮 = 水平左右平移
        elif event.modifiers() & Qt.ShiftModifier:
            h_delta = event.angleDelta().y() / 2.0 or event.angleDelta().x() / 2.0
            self._route_pan_delta(QPointF(h_delta, 0))
            event.accept()
        # 4. 普通滚轮 = 纵向上下平移
        else:
            dx = event.angleDelta().x() / 2.0
            dy = event.angleDelta().y() / 2.0
            self._route_pan_delta(QPointF(dx, dy))
            event.accept()

    def _zoom_at_cursor(self, cursor_pos: QPointF, multiplier: float):
        """单图模式：严格以光标所在物理像素点为锚点缩放，零漂移、零抖动！"""
        old_zoom = self.zoom_factor
        new_zoom = max(0.75, min(10.0, old_zoom * multiplier))
        if abs(new_zoom - old_zoom) < 1e-4:
            return

        base_sz = self._get_base_scaled_size()
        win_w, win_h = self.width(), self.height()
        if base_sz.width() <= 0 or base_sz.height() <= 0:
            return

        old_draw_w = base_sz.width() * old_zoom
        old_draw_h = base_sz.height() * old_zoom
        new_draw_w = base_sz.width() * new_zoom
        new_draw_h = base_sz.height() * new_zoom

        old_draw_x = (win_w - old_draw_w) / 2.0 + self.pan_offset.x()
        old_draw_y = (win_h - old_draw_h) / 2.0 + self.pan_offset.y()

        factor = new_zoom / old_zoom

        if new_zoom <= 1.01:
            new_pan_x = 0.0
            new_pan_y = 0.0
        else:
            new_draw_x = cursor_pos.x() - (cursor_pos.x() - old_draw_x) * factor
            new_draw_y = cursor_pos.y() - (cursor_pos.y() - old_draw_y) * factor
            new_pan_x = new_draw_x - (win_w - new_draw_w) / 2.0
            new_pan_y = new_draw_y - (win_h - new_draw_h) / 2.0

        self.zoom_factor = new_zoom
        self.pan_offset = QPointF(new_pan_x, new_pan_y)
        self._clamp_pan_offset()
        self._sync_zoom_ui()
        self._sync_minimap()

        if self.is_pure_fullscreen:
            self.minimap.setVisible(self.zoom_factor > 1.05)

        self.update()

    def _zoom_compare_at_cursor(self, pane: int, cursor_pos: QPointF, multiplier: float):
        """对比模式：以当前光标在窗格中的像素点为锚点独立缩放，绝不影响另一格！"""
        old_zoom = self._pane_zoom(pane)
        new_zoom = max(0.75, min(10.0, old_zoom * multiplier))
        if abs(new_zoom - old_zoom) < 1e-4:
            return

        pr = self._pane_rect(pane)
        base_sz = self._get_pane_base_scaled_size(pane)
        if base_sz.width() <= 0 or base_sz.height() <= 0:
            return

        old_draw_w = base_sz.width() * old_zoom
        old_draw_h = base_sz.height() * old_zoom
        new_draw_w = base_sz.width() * new_zoom
        new_draw_h = base_sz.height() * new_zoom

        pan = self.compare_pan_offset if pane == 1 else self.pan_offset
        old_draw_x = pr.x() + (pr.width() - old_draw_w) / 2.0 + pan.x()
        old_draw_y = pr.y() + (pr.height() - old_draw_h) / 2.0 + pan.y()

        factor = new_zoom / old_zoom

        if new_zoom <= 1.01:
            new_pan_x = 0.0
            new_pan_y = 0.0
        else:
            new_draw_x = cursor_pos.x() - (cursor_pos.x() - old_draw_x) * factor
            new_draw_y = cursor_pos.y() - (cursor_pos.y() - old_draw_y) * factor
            new_pan_x = new_draw_x - pr.x() - (pr.width() - new_draw_w) / 2.0
            new_pan_y = new_draw_y - pr.y() - (pr.height() - new_draw_h) / 2.0

        self._set_pane_zoom(pane, new_zoom)
        self._set_pane_pan(pane, QPointF(new_pan_x, new_pan_y))
        self._clamp_pane_pan(pane)
        self._sync_zoom_ui()
        self.update()

    # 兼容层：保障滑块与所有预设菜单正常调用
    def _zoom_at_point(self, cursor_pos: QPointF, multiplier: float):
        if self.compare_mode:
            self._zoom_compare_at_cursor(self.active_pane, cursor_pos, multiplier)
        else:
            self._zoom_at_cursor(cursor_pos, multiplier)

    def _zoom_compare_pane(self, cursor_pos: QPointF, multiplier: float):
        self._zoom_compare_at_cursor(self.active_pane, cursor_pos, multiplier)

    def _zoom_at_center(self, multiplier: float):
        center_pt = QPointF(self.width() / 2.0, self.height() / 2.0)
        self._zoom_at_cursor(center_pt, multiplier)

    def _zoom_compare_center(self, pane: int, multiplier: float):
        pr = self._pane_rect(pane)
        self._zoom_compare_at_cursor(pane, pr.center(), multiplier)

    def _on_slider_zoom_changed(self, value: int):
        """右下角滑块拖动缩放"""
        target_zoom = value / 100.0
        cur_z = self._pane_zoom(self.active_pane) if self.compare_mode else self.zoom_factor
        multiplier = target_zoom / max(cur_z, 0.01)
        if self.compare_mode:
            pr = self._pane_rect(self.active_pane)
            self._zoom_compare_at_cursor(self.active_pane, pr.center(), multiplier)
        else:
            center_pt = QPointF(self.width() / 2.0, self.height() / 2.0)
            self._zoom_at_cursor(center_pt, multiplier)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            # 优先判定预裁剪取景框手柄拖拽
            if self.crop_active and not self.compare_mode:
                base_sz = self._get_base_scaled_size()
                draw_w = base_sz.width() * self.zoom_factor
                draw_h = base_sz.height() * self.zoom_factor
                draw_x = (self.width() - draw_w) / 2.0 + self.pan_offset.x()
                draw_y = (self.height() - draw_h) / 2.0 + self.pan_offset.y()
                draw_rect = QRectF(draw_x, draw_y, draw_w, draw_h)
                crop_rect = self._get_current_crop_pixel_rect(draw_rect)

                mode = self._hit_test_crop_handle(event.position(), crop_rect)
                if mode is not None:
                    self.crop_drag_mode = mode
                    self.crop_drag_start_pos = event.position()
                    self.crop_drag_start_norm_rect = QRectF(self.norm_crop_rect)
                    event.accept()
                    return

            if self.compare_mode:
                new_pane = 1 if self._pane_rect(1).contains(event.position()) else 0
                if new_pane != self.active_pane:
                    self.active_pane = new_pane
                    self._refresh_active_pane_ui()
                    self._sync_zoom_ui()
                    self._sync_eye_zoom_ui()
                    self.sig_toggle_mark_requested.emit("__CHECK_STATE__")
                self.update()
            self.is_dragging = True
            self.last_mouse_pos = event.pos()

    def mouseMoveEvent(self, event: QMouseEvent):
        # 1. 正在拖动或拉伸裁切框
        if self.crop_active and self.crop_drag_mode and not self.compare_mode:
            base_sz = self._get_base_scaled_size()
            draw_w = base_sz.width() * self.zoom_factor
            draw_h = base_sz.height() * self.zoom_factor
            if draw_w > 0 and draw_h > 0:
                delta = event.position() - self.crop_drag_start_pos
                mode = self.crop_drag_mode
                r0 = self.crop_drag_start_norm_rect

                # 转换为屏幕物理像素进行绝对几何缩放与比例死锁
                p0_x = r0.x() * draw_w
                p0_y = r0.y() * draw_h
                p0_w = r0.width() * draw_w
                p0_h = r0.height() * draw_h

                if mode == 'move':
                    dx_norm = delta.x() / draw_w
                    dy_norm = delta.y() / draw_h
                    nx = max(0.0, min(1.0 - r0.width(), r0.x() + dx_norm))
                    ny = max(0.0, min(1.0 - r0.height(), r0.y() + dy_norm))
                    self.norm_crop_rect = QRectF(nx, ny, r0.width(), r0.height())
                else:
                    target_r = self.crop_target_aspect

                    # ==================== 分支 A: 自由比例 (解除任何长宽比约束) ====================
                    if target_r is None:
                        new_px, new_py, new_pw, new_ph = p0_x, p0_y, p0_w, p0_h
                        if 'left' in mode or mode in ('tl', 'bl'):
                            new_pw = max(40.0, p0_w - delta.x())
                            new_px = (p0_x + p0_w) - new_pw
                        elif 'right' in mode or mode in ('tr', 'br'):
                            new_pw = max(40.0, min(draw_w - p0_x, p0_w + delta.x()))

                        if 'top' in mode or mode in ('tl', 'tr'):
                            new_ph = max(40.0, p0_h - delta.y())
                            new_py = (p0_y + p0_h) - new_ph
                        elif 'bottom' in mode or mode in ('bl', 'br'):
                            new_ph = max(40.0, min(draw_h - p0_y, p0_h + delta.y()))

                    # ==================== 分支 B: 严格锁定比例 (对角锚点死锁算法) ====================
                    else:
                        if mode == 'br':
                            # 固定左上角
                            max_w = draw_w - p0_x
                            max_h = draw_h - p0_y
                            cand_w = max(40.0, p0_w + delta.x())
                            cand_h = cand_w / target_r
                            if cand_h > max_h or cand_w > max_w:
                                cand_w = min(max_w, max_h * target_r)
                                cand_h = cand_w / target_r
                            new_px, new_py, new_pw, new_ph = p0_x, p0_y, cand_w, cand_h

                        elif mode == 'tl':
                            # 固定右下角
                            anchor_r = p0_x + p0_w
                            anchor_b = p0_y + p0_h
                            cand_w = max(40.0, p0_w - delta.x())
                            cand_h = cand_w / target_r
                            if cand_w > anchor_r or cand_h > anchor_b:
                                cand_w = min(anchor_r, anchor_b * target_r)
                                cand_h = cand_w / target_r
                            new_px = anchor_r - cand_w
                            new_py = anchor_b - cand_h
                            new_pw, new_ph = cand_w, cand_h

                        elif mode == 'tr':
                            # 固定左下角
                            anchor_b = p0_y + p0_h
                            max_w = draw_w - p0_x
                            cand_w = max(40.0, p0_w + delta.x())
                            cand_h = cand_w / target_r
                            if cand_w > max_w or cand_h > anchor_b:
                                cand_w = min(max_w, anchor_b * target_r)
                                cand_h = cand_w / target_r
                            new_px = p0_x
                            new_py = anchor_b - cand_h
                            new_pw, new_ph = cand_w, cand_h

                        elif mode == 'bl':
                            # 固定右上角
                            anchor_r = p0_x + p0_w
                            max_h = draw_h - p0_y
                            cand_w = max(40.0, p0_w - delta.x())
                            cand_h = cand_w / target_r
                            if cand_w > anchor_r or cand_h > max_h:
                                cand_w = min(anchor_r, max_h * target_r)
                                cand_h = cand_w / target_r
                            new_px = anchor_r - cand_w
                            new_py = p0_y
                            new_pw, new_ph = cand_w, cand_h

                        elif mode in ('left', 'right'):
                            # 拉侧边: 水平缩放，垂直中心对称等比展开
                            cand_w = max(40.0, p0_w + (delta.x() if mode == 'right' else -delta.x()))
                            cand_h = cand_w / target_r
                            mid_y = p0_y + p0_h / 2.0
                            if cand_h > draw_h:
                                cand_h = draw_h
                                cand_w = cand_h * target_r
                            new_py = max(0.0, min(draw_h - cand_h, mid_y - cand_h / 2.0))
                            if mode == 'right':
                                cand_w = min(draw_w - p0_x, cand_w)
                                cand_h = cand_w / target_r
                                new_px = p0_x
                            else:
                                anchor_r = p0_x + p0_w
                                cand_w = min(anchor_r, cand_w)
                                cand_h = cand_w / target_r
                                new_px = anchor_r - cand_w
                            new_pw, new_ph = cand_w, cand_h

                        else:  # 'top', 'bottom'
                            # 拉底顶边: 垂直缩放，水平中心对称等比展开
                            cand_h = max(40.0, p0_h + (delta.y() if mode == 'bottom' else -delta.y()))
                            cand_w = cand_h * target_r
                            mid_x = p0_x + p0_w / 2.0
                            if cand_w > draw_w:
                                cand_w = draw_w
                                cand_h = cand_w / target_r
                            new_px = max(0.0, min(draw_w - cand_w, mid_x - cand_w / 2.0))
                            if mode == 'bottom':
                                cand_h = min(draw_h - p0_y, cand_h)
                                cand_w = cand_h * target_r
                                new_py = p0_y
                            else:
                                anchor_b = p0_y + p0_h
                                cand_h = min(anchor_b, cand_h)
                                cand_w = cand_h * target_r
                                new_py = anchor_b - cand_h
                            new_pw, new_ph = cand_w, cand_h

                    # 换算回归一化坐标并死死限制在原图内
                    nx = max(0.0, min(1.0 - (new_pw / draw_w), new_px / draw_w))
                    ny = max(0.0, min(1.0 - (new_ph / draw_h), new_py / draw_h))
                    nw = min(1.0 - nx, new_pw / draw_w)
                    nh = min(1.0 - ny, new_ph / draw_h)
                    self.norm_crop_rect = QRectF(nx, ny, nw, nh)

                self._sync_crop_ui_display()
                self.update()
                event.accept()
                return

        # 2. 鼠标悬停改变手势图标
        if self.crop_active and not self.compare_mode and not self.is_dragging:
            base_sz = self._get_base_scaled_size()
            draw_w = base_sz.width() * self.zoom_factor
            draw_h = base_sz.height() * self.zoom_factor
            draw_x = (self.width() - draw_w) / 2.0 + self.pan_offset.x()
            draw_y = (self.height() - draw_h) / 2.0 + self.pan_offset.y()
            crop_rect = self._get_current_crop_pixel_rect(QRectF(draw_x, draw_y, draw_w, draw_h))
            handle = self._hit_test_crop_handle(event.position(), crop_rect)
            if handle in ('tl', 'br'): self.setCursor(Qt.SizeFDiagCursor)
            elif handle in ('tr', 'bl'): self.setCursor(Qt.SizeBDiagCursor)
            elif handle in ('left', 'right'): self.setCursor(Qt.SizeHorCursor)
            elif handle in ('top', 'bottom'): self.setCursor(Qt.SizeVerCursor)
            elif handle == 'move': self.setCursor(Qt.SizeAllCursor)
            else: self.unsetCursor()

        if self.is_dragging:
            delta = event.pos() - self.last_mouse_pos
            self.last_mouse_pos = event.pos()
            self._route_pan_delta(QPointF(delta.x(), delta.y()))

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.is_dragging = False
            self.crop_drag_mode = None

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        self._toggle_eye_zoom()

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        sc = load_shortcuts()
        k_prev = shortcut_qt_key(sc["prev"])
        k_next = shortcut_qt_key(sc["next"])
        k_mark = shortcut_qt_key(sc["mark"])
        k_eye = shortcut_qt_key(sc["eye_zoom"])
        k_af = shortcut_qt_key(sc["af"])
        k_cmp = shortcut_qt_key(sc["compare"])
        k_grid = shortcut_qt_key(sc.get("grid", "Q"))
        k_crop = shortcut_qt_key(sc.get("precrop", "W"))

        k_cmp_l_prev = shortcut_qt_key(sc.get("cmp_l_prev", "U"))
        k_cmp_l_next = shortcut_qt_key(sc.get("cmp_l_next", "I"))
        k_cmp_r_prev = shortcut_qt_key(sc.get("cmp_r_prev", "O"))
        k_cmp_r_next = shortcut_qt_key(sc.get("cmp_r_next", "P"))

        # 【核心：双轨全键盘驾驶舱（动态读取设置绑定）】
        if key == k_cmp_l_prev:
            if self.compare_mode:
                self._pane_step(0, -1)
            else:
                self.prev_image()
            event.accept()
            return
        elif key == k_cmp_l_next:
            if self.compare_mode:
                self._pane_step(0, 1)
            else:
                self.next_image()
            event.accept()
            return
        elif key == k_cmp_r_prev:
            if self.compare_mode:
                self._pane_step(1, -1)
            event.accept()
            return
        elif key == k_cmp_r_next:
            if self.compare_mode:
                self._pane_step(1, 1)
            event.accept()
            return

        if key == Qt.Key_Escape:
            if self.is_pure_fullscreen:
                self.sig_toggle_cinema_fullscreen.emit()
            else:
                self._on_back_clicked()
            event.accept()
        # 键盘【↑ ↓ ← →】真·上下左右平移漫游！
        elif key == Qt.Key_Up:
            self._route_pan_delta(QPointF(0, 60))
            event.accept()
        elif key == Qt.Key_Down:
            self._route_pan_delta(QPointF(0, -60))
            event.accept()
        elif key == Qt.Key_Left:
            self._route_pan_delta(QPointF(60, 0))
            event.accept()
        elif key == Qt.Key_Right:
            self._route_pan_delta(QPointF(-60, 0))
            event.accept()
        elif key == k_prev:
            self.prev_image()
            event.accept()
        elif key == k_next:
            self.next_image()
            event.accept()
        elif key in (k_mark, Qt.Key_X, Qt.Key_Delete, Qt.Key_S):
            self._on_mark_toggle_clicked()
            event.accept()
        elif event.text() in ("0", "1", "2", "3", "4", "5"):
            path = self.get_active_path()
            if path:
                self.sig_rating_requested.emit(path, int(event.text()))
            event.accept()
        elif key == k_cmp:
            self._toggle_compare()
            event.accept()
        elif key == k_grid:
            if event.modifiers() & Qt.ShiftModifier:
                # Shift + Q: 翻转黄金螺旋朝向象限 (对标 Lightroom Shift+O)
                self.spiral_orientation = (self.spiral_orientation + 1) % 4
                self.update()
            else:
                self._cycle_grid_mode()
            event.accept()
        elif key == k_crop:
            self._toggle_crop_active()
            event.accept()
        # 空格键：绑定沉浸全屏 开/关！
        elif key == Qt.Key_Space:
            self._toggle_fullscreen()
            event.accept()
        # R 键：专注眼部自适应特写 开/关！
        elif key == k_eye:
            self._toggle_eye_zoom()
            event.accept()
        elif key == k_af:
            self._toggle_af_points()
            event.accept()
        else:
            super().keyPressEvent(event)

    def _sync_minimap(self):
        if self.compare_mode:
            return
        if self.current_pixmap.isNull():
            return

        base_sz = self._get_base_scaled_size()
        draw_w = base_sz.width() * self.zoom_factor
        draw_h = base_sz.height() * self.zoom_factor

        win_w, win_h = self.width(), self.height()
        draw_x = (win_w - draw_w) / 2.0 + self.pan_offset.x()
        draw_y = (win_h - draw_h) / 2.0 + self.pan_offset.y()

        vx0 = max(0.0, (-draw_x) / draw_w)
        vy0 = max(0.0, (-draw_y) / draw_h)
        vw = min(1.0, win_w / draw_w)
        vh = min(1.0, win_h / draw_h)

        norm_rect = QRectF(vx0, vy0, vw, vh)
        self.minimap.update_state(self.current_pixmap, norm_rect)

    def _on_minimap_pan(self, norm_x: float, norm_y: float):
        base_sz = self._get_base_scaled_size()
        draw_w = base_sz.width() * self.zoom_factor
        draw_h = base_sz.height() * self.zoom_factor

        win_w, win_h = self.width(), self.height()
        target_x = -(norm_x * draw_w - win_w / 2.0)
        target_y = -(norm_y * draw_h - win_h / 2.0)

        self.pan_offset = QPointF(
            target_x - (win_w - draw_w) / 2.0,
            target_y - (win_h - draw_h) / 2.0
        )
        self._clamp_pan_offset()
        self._sync_minimap()
        self.update()

    def _toggle_af_points(self):
        self.show_af_points = not self.show_af_points
        if self.show_af_points:
            self.btn_af_toggle.setStyleSheet(self._AF_BTN_ACTIVE)
        else:
            self.btn_af_toggle.setStyleSheet(self._AF_BTN_INACTIVE)
        self.update()

    def _cycle_grid_mode(self):
        """循环切换诺基亚 Lumia 构图辅助线 (0:关 -> 1:三分 -> 2:黄金比 -> 3:螺旋 -> 4:三角 -> 5:十字)"""
        self.grid_mode = (self.grid_mode + 1) % 6
        self.btn_grid_toggle.setText(self._GRID_NAMES[self.grid_mode])
        if self.grid_mode > 0:
            self.btn_grid_toggle.setStyleSheet("QPushButton { background-color: #0c4a6e; border: 1px solid #0284c7; border-radius: 2px; font-weight: 600; font-size: 11px; min-height: 20px; max-height: 20px; padding: 0 8px; color: #38bdf8; }")
        else:
            self.btn_grid_toggle.setStyleSheet("QPushButton { background-color: transparent; border: 1px solid transparent; border-radius: 2px; font-weight: 600; font-size: 11px; min-height: 20px; max-height: 20px; padding: 0 8px; color: #d4d4d8; } QPushButton:hover { background-color: #27272e; color: #38bdf8; }")
        self.update()

    @staticmethod
    def _draw_lumia_grid(painter: QPainter, rect: QRectF, mode: int):
        """相机原厂取景器纯黑构图辅助线绘制器 (与佳能/索尼/尼康机内 EVF 网格线完全一致)"""
        if mode <= 0 or rect.width() <= 10 or rect.height() <= 10:
            return

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setClipRect(rect)

        x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()

        def _draw_line(p1: QPointF, p2: QPointF):
            # 相机原厂取景线：底层 1px 极淡反差白，表层纯黑锐利实线 (死守相机质感)
            painter.setPen(QPen(QColor(255, 255, 255, 90), 2.0))
            painter.drawLine(p1, p2)
            painter.setPen(QPen(QColor(0, 0, 0, 235), 1.0))
            painter.drawLine(p1, p2)

        # 1. 经典三分线 (Rule of Thirds 3x3 + 纯黑十字准星)
        if mode == 1:
            x1, x2 = x + w / 3.0, x + w * 2.0 / 3.0
            y1, y2 = y + h / 3.0, y + h * 2.0 / 3.0
            _draw_line(QPointF(x1, y), QPointF(x1, y + h))
            _draw_line(QPointF(x2, y), QPointF(x2, y + h))
            _draw_line(QPointF(x, y1), QPointF(x + w, y1))
            _draw_line(QPointF(x, y2), QPointF(x + w, y2))
            
            # 相机原厂纯黑交点十字准星
            for ix in (x1, x2):
                for iy in (y1, y2):
                    painter.setPen(QPen(QColor(255, 255, 255, 100), 2.4))
                    painter.drawLine(QPointF(ix - 5, iy), QPointF(ix + 5, iy))
                    painter.drawLine(QPointF(ix, iy - 5), QPointF(ix, iy + 5))
                    painter.setPen(QPen(QColor(0, 0, 0, 240), 1.2))
                    painter.drawLine(QPointF(ix - 5, iy), QPointF(ix + 5, iy))
                    painter.drawLine(QPointF(ix, iy - 5), QPointF(ix, iy + 5))

        # 2. 黄金分割网格 (Phi Grid 1:0.618:1 纯黑)
        elif mode == 2:
            phi = 0.6180339887
            x1, x2 = x + w * (1.0 - phi), x + w * phi
            y1, y2 = y + h * (1.0 - phi), y + h * phi
            _draw_line(QPointF(x1, y), QPointF(x1, y + h))
            _draw_line(QPointF(x2, y), QPointF(x2, y + h))
            _draw_line(QPointF(x, y1), QPointF(x + w, y1))
            _draw_line(QPointF(x, y2), QPointF(x + w, y2))

        # 3. 经典相机黄金螺旋 (真·斐波那契连续相切海螺线 + 引导方框 + 螺旋眼准星)
        # 3. 黄金螺旋 (纯正对数螺线参数方程·一笔画连续采样，彻底杜绝任何杂线)
        elif mode == 3:
            rx, ry, rw, rh = rect.x(), rect.y(), rect.width(), rect.height()
            phi = 0.6180339887
            phi_inv = 1.6180339887

            # 1. 拟合长宽比为 1.618 的黄金矩形
            if rw >= rh:
                if rw / rh > phi_inv:
                    gh = rh
                    gw = rh * phi_inv
                else:
                    gw = rw
                    gh = rw / phi_inv
            else:
                if rh / rw > phi_inv:
                    gw = rw
                    gh = rw * phi_inv
                else:
                    gh = rh
                    gw = rh / phi_inv

            gx = rx + (rw - gw) / 2.0
            gy = ry + (rh - gh) / 2.0

            # 2. 读取翻转象限 (0:右上, 1:右下, 2:左下, 3:左上)
            viewer = painter.device()
            orient_idx = 0
            for attr in ("spiral_orientation", "_spiral_orient"):
                if hasattr(viewer, attr):
                    orient_idx = getattr(viewer, attr)
                    break

            cx, cy = gx + gw / 2.0, gy + gh / 2.0
            painter.save()
            painter.setClipRect(rect)
            painter.translate(cx, cy)
            if orient_idx == 1:
                painter.scale(1.0, -1.0)
            elif orient_idx == 2:
                painter.scale(-1.0, -1.0)
            elif orient_idx == 3:
                painter.scale(-1.0, 1.0)
            painter.translate(-cx, -cy)

            # 3. 计算黄金长方形内经典的 4 道分割正方形细虚线
            # 极点 (螺旋眼) 理论收敛坐标
            eye_x = gx + gw * (1.0 - 1.0 / (1.0 + phi ** 2))
            eye_y = gy + gh * (1.0 - phi / (1.0 + phi ** 2))

            # 绘制 4 阶正方形分割引导框 (干净极细线)
            cur_l, cur_t, cur_w, cur_h = gx, gy, gw, gh
            box_lines = []
            for s in range(5):
                if cur_w < 8 or cur_h < 8:
                    break
                if s % 4 == 0:
                    cut = cur_h if cur_w >= cur_h else cur_w * phi
                    box_lines.append((QPointF(cur_l + cut, cur_t), QPointF(cur_l + cut, cur_t + cur_h)))
                    cur_l += cut
                    cur_w -= cut
                elif s % 4 == 1:
                    cut = cur_w if cur_h >= cur_w else cur_h * phi
                    box_lines.append((QPointF(cur_l, cur_t + cut), QPointF(cur_l + cur_w, cur_t + cut)))
                    cur_t += cut
                    cur_h -= cut
                elif s % 4 == 2:
                    cut = cur_h if cur_w >= cur_h else cur_w * phi
                    box_lines.append((QPointF(cur_l + cur_w - cut, cur_t), QPointF(cur_l + cur_w - cut, cur_t + cur_h)))
                    cur_w -= cut
                elif s % 4 == 3:
                    cut = cur_w if cur_h >= cur_w else cur_h * phi
                    box_lines.append((QPointF(cur_l, cur_t + cur_h - cut), QPointF(cur_l + cur_w, cur_t + cur_h - cut)))
                    cur_h -= cut

            for p1, p2 in box_lines:
                painter.setPen(QPen(QColor(255, 255, 255, 80), 1.6))
                painter.drawLine(p1, p2)
                painter.setPen(QPen(QColor(0, 0, 0, 180), 0.9, Qt.DashLine))
                painter.drawLine(p1, p2)

            # 4. 纯连续对数螺线参数方程绘制 (一笔画采样，100% 无折角无杂线)
            # 极坐标: r = a * exp(b * theta)
            b_coeff = math.log(phi_inv) / (math.pi / 2.0)  # 约 0.30635
            max_r = math.sqrt((gx - eye_x) ** 2 + (gy + gh - eye_y) ** 2)
            theta_max = 3.2 * math.pi
            a_scale = max_r / math.exp(b_coeff * theta_max)

            spiral_path = QPainterPath()
            points_count = 180
            for i in range(points_count + 1):
                # 从最大外圈连续收缩至内圈焦点
                t = theta_max * (1.0 - (i / float(points_count)))
                r_val = a_scale * math.exp(b_coeff * t)
                # 初始相位角对准左下起笔
                theta_angle = t + math.pi * 0.75
                pt_x = eye_x + r_val * math.cos(theta_angle)
                pt_y = eye_y - r_val * math.sin(theta_angle)
                if i == 0:
                    spiral_path.moveTo(pt_x, pt_y)
                else:
                    spiral_path.lineTo(pt_x, pt_y)

            # 双层相机质感渲染
            painter.setPen(QPen(QColor(255, 255, 255, 110), 2.5))
            painter.drawPath(spiral_path)
            painter.setPen(QPen(QColor(0, 0, 0, 240), 1.2))
            painter.drawPath(spiral_path)

            # 5. 焦点 (螺旋眼) 极客准星
            painter.setPen(QPen(QColor(56, 189, 248, 220), 1.2))
            painter.drawLine(QPointF(eye_x - 6, eye_y), QPointF(eye_x + 6, eye_y))
            painter.drawLine(QPointF(eye_x, eye_y - 6), QPointF(eye_x, eye_y + 6))
            painter.drawEllipse(QPointF(eye_x, eye_y), 3.0, 3.0)

            painter.restore()

        # 4. 相机经典原厂：九宫格 + 对角线 (3x3 Grid + Diagonals，佳能/索尼取景器最经典构图线)
        elif mode == 4:
            # 1. 四角大交叉对角线 (贯穿全图的 X 轴透视引导线)
            _draw_line(QPointF(x, y), QPointF(x + w, y + h))
            _draw_line(QPointF(x + w, y), QPointF(x, y + h))

            # 2. 标准 3x3 九宫格三分线 (两横两竖卡死黄金点)
            x1, x2 = x + w / 3.0, x + w * 2.0 / 3.0
            y1, y2 = y + h / 3.0, y + h * 2.0 / 3.0
            _draw_line(QPointF(x1, y), QPointF(x1, y + h))
            _draw_line(QPointF(x2, y), QPointF(x2, y + h))
            _draw_line(QPointF(x, y1), QPointF(x + w, y1))
            _draw_line(QPointF(x, y2), QPointF(x + w, y2))

        # 5. 中心十字准星 (绝对中心纯黑靶心)
        elif mode == 5:
            cx, cy = x + w / 2.0, y + h / 2.0
            _draw_line(QPointF(cx, y), QPointF(cx, y + h))
            _draw_line(QPointF(x, cy), QPointF(x + w, cy))
            r_circle = min(w, h) * 0.10
            painter.setPen(QPen(QColor(255, 255, 255, 100), 2.2))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPointF(cx, cy), r_circle, r_circle)
            painter.setPen(QPen(QColor(0, 0, 0, 235), 1.2))
            painter.drawEllipse(QPointF(cx, cy), r_circle, r_circle)

        painter.restore() 

    def _get_live_crop_aspect_text(self) -> str:
        """
        获取当前裁切框纯整数最简约分比例文本 (图内铭牌与右上角无延迟同步双显)
        基于连分数 Farey 序列最佳有理数逼近:
        - 自动过滤 1 像素物理抖动
        - 瞬间收敛为纯整数最简约分比 (如 4:3, 16:9, 7:5, 5:4, 1:1)
        - 彻底消灭任何浮点小数点！
        """
        from fractions import Fraction

        base_sz = self._get_base_scaled_size()
        draw_w = base_sz.width() * self.zoom_factor
        draw_h = base_sz.height() * self.zoom_factor
        if draw_w <= 0 or draw_h <= 0 or self.norm_crop_rect.height() <= 0:
            return self.crop_ratio_mode.upper()

        pw = self.norm_crop_rect.width() * draw_w
        ph = self.norm_crop_rect.height() * draw_h
        if pw <= 0 or ph <= 0:
            return "1:1"

        # 1. 固定锁定比例模式：直接展示锁定标准比例 (如 4:5, 3:4, 16:9 等)
        if self.crop_target_aspect is not None and self.crop_ratio_mode != "free":
            if self.crop_ratio_mode in ("4:5", "3:4", "1:1", "9:16", "16:9"):
                return self.crop_ratio_mode
            # 自定义输入的数值 (如 2.35 或 21:9)，连分数自动求最简整数比
            frac_fixed = Fraction(self.crop_target_aspect).limit_denominator(32)
            return f"{frac_fixed.numerator}:{frac_fixed.denominator}"

        # 2. 自由拉伸模式：连分数有理数智能约分，抗抖动自动归纳最简整数比
        current_val = pw / ph
        frac_live = Fraction(current_val).limit_denominator(28)
        num, den = frac_live.numerator, frac_live.denominator

        return f"自由 {num}:{den}"

    def _sync_crop_ui_display(self):
        """同步刷新右上角实时比例徽标"""
        if self.crop_active:
            live_txt = self._get_live_crop_aspect_text()
            self.lbl_crop_live_ratio.setText(live_txt)
            self.lbl_crop_live_ratio.setVisible(True)
        else:
            self.lbl_crop_live_ratio.setVisible(False)

    def _toggle_crop_active(self):
        """开启或退出预裁剪工作台模式"""
        self.crop_active = not self.crop_active
        self.combo_crop_ratio.setVisible(self.crop_active)
        self.lbl_crop_live_ratio.setVisible(self.crop_active)
        if self.crop_active:
            self.btn_crop_toggle.setStyleSheet("QPushButton { background-color: #0c4a6e; border: 1px solid #0284c7; border-radius: 2px; font-weight: 600; font-size: 11px; min-height: 20px; max-height: 20px; padding: 0 8px; color: #38bdf8; }")
            self._apply_crop_aspect_constraint()
        else:
            self.btn_crop_toggle.setStyleSheet("QPushButton { background-color: transparent; border: 1px solid transparent; border-radius: 2px; font-weight: 600; font-size: 11px; min-height: 20px; max-height: 20px; padding: 0 8px; color: #d4d4d8; } QPushButton:hover { background-color: #27272e; color: #38bdf8; }")
            self.unsetCursor()
        self._sync_crop_ui_display()
        self.update()

    def _on_crop_ratio_changed(self):
        mode = self.combo_crop_ratio.currentData()
        if mode == "custom":
            from PySide6.QtWidgets import QInputDialog
            val, ok = QInputDialog.getText(self, "自定义裁切比例", "请输入目标宽高比 (例如 2.35:1 或 21:9):", text="2.35:1")
            if ok and val.strip():
                try:
                    parts = val.replace("：", ":").split(":")
                    if len(parts) == 2:
                        w_val, h_val = float(parts[0]), float(parts[1])
                        self.crop_target_aspect = w_val / max(0.01, h_val)
                    else:
                        self.crop_target_aspect = float(val)
                    self.crop_ratio_mode = f"{val.strip()}"
                except Exception:
                    self.crop_target_aspect = None
                    self.crop_ratio_mode = "free"
            else:
                self.combo_crop_ratio.blockSignals(True)
                self.combo_crop_ratio.setCurrentIndex(0)
                self.combo_crop_ratio.blockSignals(False)
                mode = "4:5"
                self.crop_ratio_mode = mode
                self.crop_target_aspect = 4.0 / 5.0
        else:
            self.crop_ratio_mode = mode
            aspect_map = {"4:5": 4.0/5.0, "3:4": 3.0/4.0, "1:1": 1.0, "9:16": 9.0/16.0, "16:9": 16.0/9.0, "free": None}
            self.crop_target_aspect = aspect_map.get(mode, None)

        self._apply_crop_aspect_constraint()
        self._sync_crop_ui_display()
        self.update()

    def _apply_crop_aspect_constraint(self):
        """按所选目标画幅重构裁切框初始位置 (保证框内完全在原图内)"""
        if self.crop_target_aspect is None:
            return  # 自由比例不强制锁定

        base_sz = self._get_base_scaled_size()
        if base_sz.width() <= 0 or base_sz.height() <= 0:
            return

        img_aspect = base_sz.width() / base_sz.height()
        target_aspect = self.crop_target_aspect

        # 在归一化坐标空间 (0.0~1.0) 内计算几何比例
        if img_aspect > target_aspect:
            nh = 0.88
            nw = (nh * target_aspect) / img_aspect
        else:
            nw = 0.88
            nh = (nw / target_aspect) * img_aspect

        nw = max(0.05, min(1.0, nw))
        nh = max(0.05, min(1.0, nh))
        nx = (1.0 - nw) / 2.0
        ny = (1.0 - nh) / 2.0
        self.norm_crop_rect = QRectF(nx, ny, nw, nh)

    def _get_current_crop_pixel_rect(self, draw_rect: QRectF) -> QRectF:
        """将归一化裁切框换算为当前屏幕物理像素矩形"""
        x = draw_rect.x() + self.norm_crop_rect.x() * draw_rect.width()
        y = draw_rect.y() + self.norm_crop_rect.y() * draw_rect.height()
        w = self.norm_crop_rect.width() * draw_rect.width()
        h = self.norm_crop_rect.height() * draw_rect.height()
        return QRectF(x, y, w, h)

    def _hit_test_crop_handle(self, mouse_pos: QPointF, crop_rect: QRectF) -> str:
        """测试光标落在裁切框的哪条边缘或手柄上 (8px 触控裕量)"""
        margin = 9.0
        x, y, w, h = crop_rect.x(), crop_rect.y(), crop_rect.width(), crop_rect.height()

        on_l = abs(mouse_pos.x() - x) <= margin and (y - margin <= mouse_pos.y() <= y + h + margin)
        on_r = abs(mouse_pos.x() - (x + w)) <= margin and (y - margin <= mouse_pos.y() <= y + h + margin)
        on_t = abs(mouse_pos.y() - y) <= margin and (x - margin <= mouse_pos.x() <= x + w + margin)
        on_b = abs(mouse_pos.y() - (y + h)) <= margin and (x - margin <= mouse_pos.x() <= x + w + margin)

        if on_l and on_t: return 'tl'
        if on_r and on_t: return 'tr'
        if on_l and on_b: return 'bl'
        if on_r and on_b: return 'br'
        if on_l: return 'left'
        if on_r: return 'right'
        if on_t: return 'top'
        if on_b: return 'bottom'
        if crop_rect.contains(mouse_pos): return 'move'
        return None

    def _draw_interactive_crop(self, painter: QPainter, draw_rect: QRectF):
        """专业级可交互预裁剪渲染器: 框外精准压暗 50%，框内 100% 原图 + 专属纯黑九宫格"""
        if not self.crop_active or draw_rect.width() <= 10 or draw_rect.height() <= 10:
            return

        crop_rect = self._get_current_crop_pixel_rect(draw_rect)

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setClipRect(draw_rect)

        # 1. 框外 50% 纯黑遮罩 (利用差集精准避开框内，框内 100% 原始透光)
        mask_path = QPainterPath()
        mask_path.addRect(draw_rect)
        mask_path.addRect(crop_rect)
        painter.setBrush(QColor(0, 0, 0, 128))
        painter.setPen(Qt.NoPen)
        painter.drawPath(mask_path)

        # 2. 框内专属九宫格三分辅助线 (纯黑相机取景线 + 极细反差底)
        cx, cy, cw, ch = crop_rect.x(), crop_rect.y(), crop_rect.width(), crop_rect.height()
        x1, x2 = cx + cw / 3.0, cx + cw * 2.0 / 3.0
        y1, y2 = cy + ch / 3.0, cy + ch * 2.0 / 3.0

        def _draw_grid_line(p1, p2):
            painter.setPen(QPen(QColor(255, 255, 255, 75), 1.8))
            painter.drawLine(p1, p2)
            painter.setPen(QPen(QColor(0, 0, 0, 200), 1.0))
            painter.drawLine(p1, p2)

        _draw_grid_line(QPointF(x1, cy), QPointF(x1, cy + ch))
        _draw_grid_line(QPointF(x2, cy), QPointF(x2, cy + ch))
        _draw_grid_line(QPointF(cx, y1), QPointF(cx + cw, y1))
        _draw_grid_line(QPointF(cx, y2), QPointF(cx + cw, y2))

        # 3. 裁切取景框外缘实线 + 四角战术直角手柄
        painter.setPen(QPen(QColor(255, 255, 255, 140), 1.0))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(crop_rect)

        # 强化四角粗手柄 (明示可拖动缩放)
        handle_len = min(18.0, cw * 0.15, ch * 0.15)
        painter.setPen(QPen(QColor(56, 189, 248), 2.5))
        # Top-Left
        painter.drawLine(QPointF(cx, cy), QPointF(cx + handle_len, cy))
        painter.drawLine(QPointF(cx, cy), QPointF(cx, cy + handle_len))
        # Top-Right
        painter.drawLine(QPointF(cx + cw, cy), QPointF(cx + cw - handle_len, cy))
        painter.drawLine(QPointF(cx + cw, cy), QPointF(cx + cw, cy + handle_len))
        # Bottom-Left
        painter.drawLine(QPointF(cx, cy + ch), QPointF(cx + handle_len, cy + ch))
        painter.drawLine(QPointF(cx, cy + ch), QPointF(cx, cy + ch - handle_len))
        # Bottom-Right
        painter.drawLine(QPointF(cx + cw, cy + ch), QPointF(cx + cw - handle_len, cy + ch))
        painter.drawLine(QPointF(cx + cw, cy + ch), QPointF(cx + cw, cy + ch - handle_len))

        # 4. 左上角比例铭牌 (实时跟随当前真实动态比值)
        tag_str = self._get_live_crop_aspect_text()
        font = QFont("Consolas", 10, QFont.Bold)
        painter.setFont(font)
        fm = QFontMetrics(font)
        tw = fm.horizontalAdvance(tag_str) + 10
        tag_box = QRectF(cx + 4, cy + 4, tw, 19)
        painter.fillRect(tag_box, QColor(0, 0, 0, 200))
        painter.setPen(QPen(QColor(56, 189, 248, 160), 1.0))
        painter.drawRect(tag_box)
        painter.setPen(QColor(56, 189, 248))
        painter.drawText(tag_box, Qt.AlignCenter, tag_str)

        painter.restore()
        

# ==============================================================================
# 暗房一键导出后台工作线程 (支持开关、自定义文件夹名、SMB 分流)
# ==============================================================================
class DarkroomExportWorker(QThread):
    sig_progress = Signal(int, int, str)
    sig_finished = Signal(dict)

    def __init__(self, all_photos: list, unhidden_photos: list, dest_dir: Path, raw_src_dir: Path,
                 config: dict, smb_dest_dir: Path = None):
        super().__init__()
        self.all_photos = all_photos
        self.unhidden_photos = unhidden_photos
        self.dest_dir = dest_dir
        self.raw_src_dir = raw_src_dir
        self.config = config
        self.smb_dest_dir = smb_dest_dir

    @staticmethod
    def normalize_stem(stem: str) -> str:
        lower = stem.lower()
        if lower.startswith("img_") or lower.startswith("_mg_"):
            return lower[4:]
        if lower.startswith("dsc0"):
            return lower[4:]
        if lower.startswith("_dsc"):
            return lower[4:]
        return lower

    def run(self):
        start_time = time.time()

        enable_all = self.config.get("enable_all_jpg", True)
        name_all = self.config.get("name_all_jpg", "jpg").strip() or "jpg"
        enable_sel = self.config.get("enable_selected_jpg", True)
        name_sel = self.config.get("name_selected_jpg", "jpg2").strip() or "jpg2"
        enable_raw = self.config.get("enable_raw", True)
        name_raw = self.config.get("name_raw", "cr3").strip() or "cr3"
        enable_smb = bool(self.smb_dest_dir)

        jpg_dir = self.dest_dir / name_all if enable_all else None
        jpg2_dir = self.dest_dir / name_sel if enable_sel else None
        cr3_dir = self.dest_dir / name_raw if enable_raw else None
        smb_cr3_dir = self.smb_dest_dir / name_raw if enable_smb else None

        if jpg_dir: jpg_dir.mkdir(parents=True, exist_ok=True)
        if jpg2_dir: jpg2_dir.mkdir(parents=True, exist_ok=True)
        if cr3_dir: cr3_dir.mkdir(parents=True, exist_ok=True)
        if smb_cr3_dir:
            try:
                smb_cr3_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                smb_cr3_dir = None

        total_steps = 0
        if enable_all: total_steps += len(self.all_photos)
        if enable_sel: total_steps += len(self.unhidden_photos)
        if enable_raw or smb_cr3_dir: total_steps += len(self.unhidden_photos)

        cur_step = 0
        skipped_jpg_all = 0
        skipped_jpg2 = 0

        def _deliver_jpg(src: Path, dst_dir: Path, cur: int, total: int, tag: str) -> bool:
            if is_raw_file(src):
                img = load_photo_qimage(str(src))
                if img.isNull():
                    self.sig_progress.emit(cur, total, f"跳过 RAW (无内嵌预览): {src.name}")
                    return False
                out = dst_dir / (src.stem + ".jpg")
                try:
                    if not img.save(str(out), "JPG", 92):
                        return False
                except Exception:
                    return False
                self.sig_progress.emit(cur, total, f"{tag} (RAW转预览): {out.name}")
                return True
            self.sig_progress.emit(cur, total, f"{tag}: {src.name}")
            shutil.copy2(src, dst_dir / src.name)
            return True

        if enable_all and jpg_dir:
            for p_str in self.all_photos:
                src = Path(p_str)
                cur_step += 1
                if not _deliver_jpg(src, jpg_dir, cur_step, total_steps, f"导出到 {name_all}"):
                    skipped_jpg_all += 1

        if enable_sel and jpg2_dir:
            for p_str in self.unhidden_photos:
                src = Path(p_str)
                cur_step += 1
                if not _deliver_jpg(src, jpg2_dir, cur_step, total_steps, f"导出到 {name_sel}"):
                    skipped_jpg2 += 1

        cr3_copied_count = 0
        smb_copied_count = 0
        orphan_count = 0

        if enable_raw or smb_cr3_dir:
            raw_index = {}
            if self.raw_src_dir.exists() and self.raw_src_dir.is_dir():
                for f in self.raw_src_dir.iterdir():
                    if f.is_file() and f.suffix.lower() in {".cr3", ".arw"}:
                        key = self.normalize_stem(f.stem)
                        raw_index[key] = f

            for p_str in self.unhidden_photos:
                src = Path(p_str)
                cur_step += 1
                if is_raw_file(src):
                    raw_file = src
                else:
                    raw_file = raw_index.get(self.normalize_stem(src.stem))

                if raw_file:
                    self.sig_progress.emit(cur_step, total_steps, f"匹配生肉 RAW: {raw_file.name}")
                    if cr3_dir:
                        shutil.copy2(raw_file, cr3_dir / raw_file.name)
                        cr3_copied_count += 1
                    if smb_cr3_dir:
                        try:
                            shutil.copy2(raw_file, smb_cr3_dir / raw_file.name)
                            smb_copied_count += 1
                        except Exception:
                            pass
                else:
                    orphan_count += 1

        elapsed = max(0.1, time.time() - start_time)
        self.sig_finished.emit({
            "dest_dir": str(self.dest_dir),
            "smb_dest_dir": str(self.smb_dest_dir) if smb_cr3_dir else "",
            "enable_all": enable_all,
            "enable_sel": enable_sel,
            "enable_raw": enable_raw,
            "enable_smb": bool(smb_cr3_dir),
            "name_all": name_all,
            "name_sel": name_sel,
            "name_raw": name_raw,
            "jpg_count": (len(self.all_photos) - skipped_jpg_all) if enable_all else 0,
            "jpg2_count": (len(self.unhidden_photos) - skipped_jpg2) if enable_sel else 0,
            "cr3_count": cr3_copied_count,
            "smb_count": smb_copied_count,
            "orphan_count": orphan_count,
            "skipped_jpg": skipped_jpg_all + skipped_jpg2,
            "elapsed": elapsed
        })

# ==============================================================================
# 导出成功结果对话框 (支持一键原生打开本地 / 移动端目录)
# ==============================================================================
class ExportSuccessDialog(QDialog):
    def __init__(self, res: dict, parent=None):
        super().__init__(parent)
        self.res = res
        self.setWindowTitle("🎉 导出交付完成")
        self.resize(520, 320)
        self.setStyleSheet("""
            QDialog { background-color: #141418; color: #f4f4f5; }
            QLabel { color: #e4e4e7; font-size: 13px; }
            QPushButton {
                background-color: #27272a;
                border: 1px solid #3f3f46;
                border-radius: 6px;
                padding: 7px 16px;
                font-weight: bold;
                color: #ffffff;
            }
            QPushButton:hover { background-color: #3f3f46; border-color: #38bdf8; color: #38bdf8; }
            QPushButton#PrimaryBtn { background-color: #0284c7; border: none; }
            QPushButton#PrimaryBtn:hover { background-color: #0369a1; color: #ffffff; }
        """)
        self._init_ui()

    def _init_ui(self):
        v = QVBoxLayout(self)
        v.setContentsMargins(20, 20, 20, 20)
        v.setSpacing(14)

        title = QLabel("✅ 选片交付任务已全自动执行完毕！")
        title.setStyleSheet("font-size: 15px; font-weight: bold; color: #38bdf8;")
        v.addWidget(title)

        lines = []
        if self.res.get("enable_all"):
            lines.append(f"• 全量暗房 [{self.res['name_all']}]: 导出 {self.res['jpg_count']} 张")
        if self.res.get("enable_sel"):
            lines.append(f"• 精选出片 [{self.res['name_sel']}]: 导出 {self.res['jpg2_count']} 张")
        if self.res.get("enable_raw"):
            lines.append(f"• 生肉归档 [{self.res['name_raw']}]: 匹配写入 {self.res['cr3_count']} 张")
        if self.res.get("enable_smb"):
            lines.append(f"• 移动端修图分流: 同步投送 {self.res['smb_count']} 张纯 RAW")
        if self.res.get("orphan_count", 0) > 0:
            lines.append(f"• ⚠️ 缺失 RAW 孤儿图: {self.res['orphan_count']} 张")
        if self.res.get("skipped_jpg", 0) > 0:
            lines.append(f"• 跳过无预览 RAW: {self.res['skipped_jpg']} 张")
        lines.append(f"• 总耗时: {self.res['elapsed']:.2f} 秒")

        lbl_summary = QLabel("\n".join(lines))
        lbl_summary.setStyleSheet("background-color: #18181e; border: 1px solid #27272f; border-radius: 6px; padding: 12px; font-family: Consolas, 'Segoe UI', monospace; line-height: 1.5;")
        v.addWidget(lbl_summary)

        v.addStretch()

        btns = QHBoxLayout()
        btns.setSpacing(10)

        btn_open_local = QPushButton("📁 打开本地导出目录")
        btn_open_local.setObjectName("PrimaryBtn")
        btn_open_local.clicked.connect(self._open_local_dir)
        btns.addWidget(btn_open_local)

        if self.res.get("smb_dest_dir"):
            btn_open_remote = QPushButton("📱 打开移动端共享目录")
            btn_open_remote.clicked.connect(self._open_remote_dir)
            btns.addWidget(btn_open_remote)

        btns.addStretch()

        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(self.accept)
        btns.addWidget(btn_close)

        v.addLayout(btns)

    def _open_local_dir(self):
        d = self.res.get("dest_dir", "")
        if d and os.path.exists(d):
            try:
                os.startfile(d)
            except Exception:
                from PySide6.QtGui import QDesktopServices
                from PySide6.QtCore import QUrl
                QDesktopServices.openUrl(QUrl.fromLocalFile(d))

    def _open_remote_dir(self):
        d = self.res.get("smb_dest_dir", "")
        if d and os.path.exists(d):
            try:
                os.startfile(d)
            except Exception:
                from PySide6.QtGui import QDesktopServices
                from PySide6.QtCore import QUrl
                QDesktopServices.openUrl(QUrl.fromLocalFile(d))


# ==============================================================================
# 暗房一键导出配置对话框 (自动建目录 + 根目录二合一 + 移动端修图分流)
# ==============================================================================
class ExportDialog(QDialog):
    def __init__(self, parent=None, default_work_dir=""):
        super().__init__(parent)
        self.setWindowTitle("暗房一键导出交付 (自动建目录 + 移动端修图闭环)")
        self.resize(580, 420)
        self.setStyleSheet("""
            QDialog { background-color: #18181b; color: #f4f4f5; }
            QLabel { color: #e4e4e7; font-weight: bold; font-size: 12px; }
            QLineEdit { background-color: #27272a; border: 1px solid #3f3f46; border-radius: 6px; padding: 6px; color: #fff; font-size: 12px; }
            QLineEdit:focus { border: 1px solid #38bdf8; background-color: #202028; }
            QPushButton { background-color: #27272a; border: 1px solid #3f3f46; border-radius: 6px; padding: 6px 14px; font-weight: bold; }
            QPushButton:hover { background-color: #3f3f46; border-color: #38bdf8; }
            QPushButton#ActionBtn { background-color: #0284c7; color: white; border: none; padding: 8px 22px; font-size: 13px; }
            QPushButton#ActionBtn:hover { background-color: #0369a1; }
            QCheckBox { color: #38bdf8; font-weight: bold; }
        """)

        self.default_work_dir = default_work_dir
        self.st = QSettings("AnicropGeeks", "AnicropStudio")
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(18, 18, 18, 18)

        # 1. 角色名（流萤等二次元角色）
        layout.addWidget(QLabel("🏷️ 1. 角色文件夹名（或者coser老师的cv）:"))
        self.edit_task_name = QLineEdit()
        self.edit_task_name.setPlaceholderText("例如: 流萤")
        self.edit_task_name.textChanged.connect(self._update_preview)
        layout.addWidget(self.edit_task_name)

        # 2. 本地工作根目录 (2和3完美二合一：既是 RAW 源，也是导出分类根目录)
        layout.addWidget(QLabel("📁 2. 本地工作根目录 (既包含全量 RAW 素材，也在此自动生成角色文件夹):"))
        h_work = QHBoxLayout()
        saved_work_dir = self.default_work_dir or self.st.value("last_work_root_dir", "")
        self.edit_work_root = QLineEdit(saved_work_dir)
        self.edit_work_root.setPlaceholderText("选择当前活动的总文件夹，如 E:\\Photos\\Bilibili World 2026")
        self.edit_work_root.textChanged.connect(self._update_preview)
        h_work.addWidget(self.edit_work_root)
        btn_b_work = QPushButton("浏览...")
        btn_b_work.clicked.connect(self._browse_work_root)
        h_work.addWidget(btn_b_work)
        layout.addLayout(h_work)

        # 本地生成路径动态预览
        self.lbl_preview_local = QLabel("👉 本地输出预览: 请输入角色名称")
        self.lbl_preview_local.setStyleSheet("color: #38bdf8; font-size: 11px; font-family: Consolas, monospace;")
        layout.addWidget(self.lbl_preview_local)

        # 3. 移动端/iPad 修图路径 (本机作为服务机，直接投送到给 iPad 共享的物理路径)
        remote_box = QFrame()
        remote_box.setStyleSheet("QFrame { background-color: #141418; border: 1px solid #27272f; border-radius: 8px; padding: 10px; }")
        remote_vlayout = QVBoxLayout(remote_box)
        remote_vlayout.setSpacing(8)

        self.cb_enable_remote = QCheckBox("📱 启用异地/移动端修图分流 (自动投送纯 RAW 供 iPad 像素蛋糕秒开)")
        saved_remote_enabled = self.st.value("export_enable_remote", False, type=bool)
        self.cb_enable_remote.setChecked(saved_remote_enabled)
        self.cb_enable_remote.toggled.connect(self._on_remote_toggled)
        remote_vlayout.addWidget(self.cb_enable_remote)

        h_remote = QHBoxLayout()
        saved_remote_path = self.st.value("export_remote_path", "")
        self.edit_remote_path = QLineEdit(saved_remote_path)
        self.edit_remote_path.setPlaceholderText("选择本机共享给 iPad 的修图根目录，如 D:\\iPad_Share")
        self.edit_remote_path.textChanged.connect(self._update_preview)
        h_remote.addWidget(self.edit_remote_path)
        self.btn_b_remote = QPushButton("浏览...")
        self.btn_b_remote.clicked.connect(self._browse_remote_path)
        h_remote.addWidget(self.btn_b_remote)
        remote_vlayout.addLayout(h_remote)

        self.lbl_preview_remote = QLabel("👉 移动端同步预览: 未启用")
        self.lbl_preview_remote.setStyleSheet("color: #a1a1aa; font-size: 11px; font-family: Consolas, monospace;")
        remote_vlayout.addWidget(self.lbl_preview_remote)
        layout.addWidget(remote_box)

        layout.addStretch()

        btns = QHBoxLayout()
        btns.addStretch()
        btn_cancel = QPushButton("取消")
        btn_cancel.clicked.connect(self.reject)
        btns.addWidget(btn_cancel)

        btn_ok = QPushButton("⚡ 开始一键导出")
        btn_ok.setObjectName("ActionBtn")
        btn_ok.clicked.connect(self._on_ok)
        btns.addWidget(btn_ok)
        layout.addLayout(btns)

        self._on_remote_toggled(self.cb_enable_remote.isChecked())
        self._update_preview()

    def _browse_work_root(self):
        d = QFileDialog.getExistingDirectory(self, "选择本地工作根目录", self.edit_work_root.text() or "")
        if d:
            self.edit_work_root.setText(d)

    def _browse_remote_path(self):
        d = QFileDialog.getExistingDirectory(self, "选择移动端修图根目录", self.edit_remote_path.text() or "")
        if d:
            self.edit_remote_path.setText(d)

    def _on_remote_toggled(self, checked: bool):
        self.edit_remote_path.setEnabled(checked)
        self.btn_b_remote.setEnabled(checked)
        self._update_preview()

    def _update_preview(self):
        task = self.edit_task_name.text().strip() or "[角色名称]"
        work_root = self.edit_work_root.text().strip()
        if work_root:
            p_local = Path(work_root) / task
            self.lbl_preview_local.setText(f"👉 本地将自动创建: {p_local}")
            self.lbl_preview_local.setStyleSheet("color: #38bdf8; font-size: 11px; font-family: Consolas, monospace;")
        else:
            self.lbl_preview_local.setText("👉 本地输出预览: 请指定工作根目录")
            self.lbl_preview_local.setStyleSheet("color: #facc15; font-size: 11px;")

        if self.cb_enable_remote.isChecked():
            remote_root = self.edit_remote_path.text().strip()
            if remote_root:
                p_remote = Path(remote_root) / task
                self.lbl_preview_remote.setText(f"👉 移动端将同步创建: {p_remote}")
                self.lbl_preview_remote.setStyleSheet("color: #4ade80; font-size: 11px; font-family: Consolas, monospace;")
            else:
                self.lbl_preview_remote.setText("👉 移动端同步预览: 请指定移动端修图目录")
                self.lbl_preview_remote.setStyleSheet("color: #facc15; font-size: 11px;")
        else:
            self.lbl_preview_remote.setText("👉 移动端同步预览: 未启用")
            self.lbl_preview_remote.setStyleSheet("color: #71717a; font-size: 11px;")

    def _on_ok(self):
        task = self.edit_task_name.text().strip()
        work_root = self.edit_work_root.text().strip()

        if not task:
            QMessageBox.warning(self, "名称缺失", "请输入角色文件夹名！")
            return
        if not work_root or not Path(work_root).exists():
            QMessageBox.warning(self, "目录无效", "请指定有效的工作根目录！")
            return

        if self.cb_enable_remote.isChecked():
            remote_root = self.edit_remote_path.text().strip()
            if not remote_root or not Path(remote_root).exists():
                QMessageBox.warning(self, "目录无效", "启用了移动端同步，但移动端修图目录不存在！")
                return

        # 持久化记忆
        self.st.setValue("last_work_root_dir", work_root)
        self.st.setValue("export_enable_remote", self.cb_enable_remote.isChecked())
        if self.cb_enable_remote.isChecked():
            self.st.setValue("export_remote_path", self.edit_remote_path.text().strip())

        # 核心：dest_dir 与 raw_src_dir 逻辑闭环
        self.final_dest_dir = Path(work_root) / task
        self.final_raw_src_dir = Path(work_root)
        self.final_smb_dest_dir = (Path(self.edit_remote_path.text().strip()) / task) if self.cb_enable_remote.isChecked() else None
        self.accept()

# ==============================================================================
# P0.2 漫展画廊相册组件
# ==============================================================================
class AlbumPageWidget(QWidget):
    image_exts = VALID_IMAGE_EXTS        # 子类覆盖即可变身 RAW 专属画廊
    settings_dir_key = "last_album_dir"  # 子类覆盖，避免和主画廊互相顶掉记忆路径
    sig_selection_changed = Signal(str, int, int)
    sig_toggle_cinema_fullscreen = Signal()
    sig_enter_darkroom = Signal(list, dict, str)
    sig_viewer_active = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_dir = ""
        self.photos = []
        self._all_photos_found = []
        self._all_jpgs = []
        self._all_raws = []
        self._sub_dirs = []
        self.ratings = {}
        self.model = PhotoListModel()
        self.active_cache_dir = None
        self._live_thumb_tasks = set()   # QRunnable 存续保险（防 autoDelete 与 GC 双重析构）
        self.warmup_total = 0
        self.warmup_finished_count = 0

        self.nav_history = []
        self.nav_index = -1

        self._init_ui()
        self._load_initial_default_path()

    def _init_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.page_stack = QStackedWidget()

        self.gallery_container = QWidget()
        g_layout = QVBoxLayout(self.gallery_container)
        g_layout.setContentsMargins(0, 0, 0, 0)
        g_layout.setSpacing(6)

        # 【Windows 11 资源管理器风格：全宽独立双行面板】
        self.header_widget = QFrame()
        self.header_widget.setObjectName("ExplorerToolbar")
        self.header_widget.setStyleSheet("""
            QFrame#ExplorerToolbar {
                background-color: #141418;
                border-bottom: 1px solid #27272f;
                padding: 4px 6px;
            }
        """)
        h_vlayout = QVBoxLayout(self.header_widget)
        h_vlayout.setContentsMargins(6, 4, 6, 4)
        h_vlayout.setSpacing(6)

        btn_nav_style = """
            QPushButton {
                background-color: #27272a;
                border: 1px solid #3f3f46;
                border-radius: 4px;
                padding: 4px 10px;
                font-weight: bold;
                color: #e4e4e7;
            }
            QPushButton:hover { background-color: #3f3f46; color: #ffffff; }
            QPushButton:disabled { background-color: #18181b; color: #52525b; border-color: #27272a; }
        """

        # 第一行：全宽物理地址导航行 (大图检视时整行隐藏)
        self.row1_widget = QWidget()
        row1 = QHBoxLayout(self.row1_widget)
        row1.setContentsMargins(0, 0, 0, 0)
        row1.setSpacing(6)

        self.btn_nav_back = QPushButton("←")
        self.btn_nav_back.setStyleSheet(btn_nav_style)
        self.btn_nav_back.setToolTip("后退")
        self.btn_nav_back.setEnabled(False)
        self.btn_nav_back.clicked.connect(self._nav_go_back)
        row1.addWidget(self.btn_nav_back)

        self.btn_nav_forward = QPushButton("→")
        self.btn_nav_forward.setStyleSheet(btn_nav_style)
        self.btn_nav_forward.setToolTip("前进")
        self.btn_nav_forward.setEnabled(False)
        self.btn_nav_forward.clicked.connect(self._nav_go_forward)
        row1.addWidget(self.btn_nav_forward)

        self.btn_nav_up = QPushButton("↑")
        self.btn_nav_up.setStyleSheet(btn_nav_style)
        self.btn_nav_up.setToolTip("向上一级")
        self.btn_nav_up.clicked.connect(self._nav_go_up)
        row1.addWidget(self.btn_nav_up)

        self.edit_address = QLineEdit()
        self.edit_address.setPlaceholderText("输入或粘贴物理路径，回车秒级跳转...")
        self.edit_address.setStyleSheet("QLineEdit { background-color: #121214; border: 1px solid #3f3f46; border-radius: 4px; padding: 5px 8px; color: #38bdf8; font-family: 'Segoe UI', sans-serif; font-size: 12px; } QLineEdit:focus { border: 1px solid #38bdf8; }")
        self.edit_address.returnPressed.connect(self._on_address_entered)
        row1.addWidget(self.edit_address, stretch=1)

        self.btn_open = QPushButton("📂 浏览...")
        self.btn_open.setStyleSheet(btn_nav_style)
        self.btn_open.clicked.connect(self._browse_folder)
        row1.addWidget(self.btn_open)

        h_vlayout.addWidget(self.row1_widget)

        # 第二行：双模式菜单栏容器 (画廊模式 / 检视器模式 二选一)
        self.row2_widget = QWidget()
        row2 = QHBoxLayout(self.row2_widget)
        row2.setContentsMargins(0, 0, 0, 0)
        row2.setSpacing(8)

        # 模式 A: 画廊专属控制行
        self.row2_gallery_box = QWidget()
        row2_gal = QHBoxLayout(self.row2_gallery_box)
        row2_gal.setContentsMargins(0, 0, 0, 0)
        row2_gal.setSpacing(8)

        self.btn_select_mode = QPushButton("☑ 选择照片")
        self.btn_select_mode.setStyleSheet("QPushButton { background-color: #27272a; border: 1px solid #3f3f46; border-radius: 4px; padding: 4px 12px; font-weight: bold; } QPushButton:hover { background-color: #3f3f46; }")
        self.btn_select_mode.clicked.connect(self._toggle_selection_mode)
        row2_gal.addWidget(self.btn_select_mode)

        lbl_fmt = QLabel("格式:")
        lbl_fmt.setStyleSheet("color: #a1a1aa; font-size: 12px;")
        row2_gal.addWidget(lbl_fmt)

        self.combo_format = QComboBox()
        self.combo_format.setMinimumWidth(125)
        self.combo_format.addItem("📷 仅 JPG 照片", "jpg")
        self.combo_format.addItem("🎞️ 仅 RAW 格式", "raw")
        self.combo_format.addItem("🔀 全部格式 (混合)", "all")
        self.combo_format.currentIndexChanged.connect(self._on_sort_or_filter_changed)
        row2_gal.addWidget(self.combo_format)

        lbl_sort = QLabel("排序:")
        lbl_sort.setStyleSheet("color: #a1a1aa; font-size: 12px;")
        row2_gal.addWidget(lbl_sort)

        self.combo_sort = QComboBox()
        self.combo_sort.setMinimumWidth(150)
        self.combo_sort.addItem("文件名 (A→Z)", "name")
        self.combo_sort.addItem("拍摄时间 (新→旧)", "mtime_desc")
        self.combo_sort.addItem("拍摄时间 (旧→新)", "mtime_asc")
        self.combo_sort.addItem("文件大小 (大→小)", "size_desc")
        self.combo_sort.addItem("⭐ 打星优先 (高→低)", "rating")
        self.combo_sort.currentIndexChanged.connect(self._on_sort_or_filter_changed)
        row2_gal.addWidget(self.combo_sort)

        self.combo_filter = QComboBox()
        self.combo_filter.setMinimumWidth(105)
        self.combo_filter.addItem("全部照片", "all")
        self.combo_filter.addItem("仅已打星 ⭐", "rated")
        self.combo_filter.addItem("仅未打星", "unrated")
        self.combo_filter.currentIndexChanged.connect(self._on_sort_or_filter_changed)
        row2_gal.addWidget(self.combo_filter)

        row2_gal.addStretch()

        self.lbl_stats = QLabel("素材: 0 项")
        self.lbl_stats.setStyleSheet("color: #a1a1aa; padding-right: 4px; font-weight: bold;")
        row2_gal.addWidget(self.lbl_stats)

        row2.addWidget(self.row2_gallery_box)

# 模式 B: 大图检视器专属顶部菜单栏 (36px 紧凑极客细轨)
        self.row2_viewer_box = QWidget()
        self.row2_viewer_box.setVisible(False)
        self.row2_viewer_box.setFixedHeight(36)
        self.row2_viewer_layout = QHBoxLayout(self.row2_viewer_box)
        self.row2_viewer_layout.setContentsMargins(0, 0, 6, 0)
        self.row2_viewer_layout.setSpacing(0)
        row2.addWidget(self.row2_viewer_box, stretch=1)

        h_vlayout.addWidget(self.row2_widget)

        settings = QSettings("AnicropGeeks", "AnicropStudio")
        self.combo_format.blockSignals(True)
        idx = self.combo_format.findData(settings.value("album_format_mode", "jpg"))
        self.combo_format.setCurrentIndex(max(0, idx))
        self.combo_format.blockSignals(False)

        self.combo_sort.blockSignals(True)
        idx = self.combo_sort.findData(settings.value("album_sort_mode", "name"))
        self.combo_sort.setCurrentIndex(max(0, idx))
        self.combo_sort.blockSignals(False)
        self.combo_filter.blockSignals(True)
        idx = self.combo_filter.findData(settings.value("album_filter_mode", "all"))
        self.combo_filter.setCurrentIndex(max(0, idx))
        self.combo_filter.blockSignals(False)

        # 【核心实例】画廊网格视图
        self.gallery = AdvancedGalleryView(self, is_darkroom=False)
        self.gallery.setModel(self.model)
        self.gallery.delegate.ratings = self.ratings
        self.gallery.doubleClicked.connect(self._on_gallery_double_clicked)
        self.gallery.selectionModel().currentChanged.connect(self._on_current_changed)
        self.gallery.sig_zoom_requested.connect(self._on_grid_zoom_step)
        self.gallery.sig_selection_state_changed.connect(self._on_selection_updated)
        g_layout.addWidget(self.gallery, stretch=1)

        # 底部控制条
        bottom_bar = QFrame()
        bottom_bar.setStyleSheet("background-color: #18181b; border-radius: 6px; border: 1px solid #27272a;")
        b_layout = QHBoxLayout(bottom_bar)
        b_layout.setContentsMargins(12, 4, 12, 4)

        lbl_hint = QLabel("💡 提示：双击进入大图检视；【R】特写；【Q】构图辅助；【W】裁切；【0~5】打星；【H】选入暗房")
        lbl_hint.setStyleSheet("color: #71717a; font-size: 11px;")
        b_layout.addWidget(lbl_hint)

        b_layout.addStretch()

        self.btn_enter_darkroom = QPushButton("🧪 进入暗房 (已选 0 张)  ➔", bottom_bar)
        self.btn_enter_darkroom.setStyleSheet("""
            QPushButton {
                background-color: #0284c7;
                color: white;
                font-weight: bold;
                font-size: 12px;
                border: none;
                border-radius: 12px;
                padding: 4px 18px;
            }
            QPushButton:hover { background-color: #0369a1; }
        """)
        self.btn_enter_darkroom.setVisible(False)
        self.btn_enter_darkroom.clicked.connect(self._on_enter_darkroom_clicked)
        b_layout.addWidget(self.btn_enter_darkroom)

        b_layout.addStretch()

        self.lbl_slider_val = QLabel("缩略图: 100%", bottom_bar)
        self.lbl_slider_val.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 12px;")
        b_layout.addWidget(self.lbl_slider_val)

        self.slider_grid_size = QSlider(Qt.Horizontal, bottom_bar)
        self.slider_grid_size.setRange(130, 420)
        self.slider_grid_size.setValue(220)
        self.slider_grid_size.setFixedWidth(140)
        self.slider_grid_size.valueChanged.connect(self._on_grid_slider_changed)
        b_layout.addWidget(self.slider_grid_size)

        g_layout.addWidget(bottom_bar)
        self.page_stack.addWidget(self.gallery_container)

        self.loading_box = QFrame()
        self.loading_box.setStyleSheet("background-color: #121214;")
        load_layout = QVBoxLayout(self.loading_box)
        load_layout.setAlignment(Qt.AlignCenter)
        load_layout.setSpacing(14)

        self.lbl_loading_title = QLabel("⚡ 正在极速预热超清 512px 缩略图...")
        self.lbl_loading_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #38bdf8;")
        load_layout.addWidget(self.lbl_loading_title, alignment=Qt.AlignCenter)

        self.progress_loading = QProgressBar()
        self.progress_loading.setFixedWidth(460)
        self.progress_loading.setValue(0)
        load_layout.addWidget(self.progress_loading, alignment=Qt.AlignCenter)

        self.lbl_loading_detail = QLabel("已就绪: 0 / 0 (0%)")
        self.lbl_loading_detail.setStyleSheet("color: #a1a1aa; font-family: monospace;")
        load_layout.addWidget(self.lbl_loading_detail, alignment=Qt.AlignCenter)
        self.page_stack.addWidget(self.loading_box)

        # 单张大图检视器
        self.photo_viewer = SinglePhotoViewer(self, is_darkroom=False)
        self.photo_viewer.sig_back_to_gallery.connect(self._on_viewer_back)
        self.photo_viewer.sig_toggle_cinema_fullscreen.connect(self.sig_toggle_cinema_fullscreen)
        self.photo_viewer.sig_toggle_mark_requested.connect(self._on_viewer_toggle_select)
        self.photo_viewer.sig_photo_changed.connect(self.sig_selection_changed)
        self.photo_viewer.sig_rating_requested.connect(self._on_viewer_rating_requested)

        # 左侧：正方形撑满高度的 "<" 返回按键
        self.row2_viewer_layout.addWidget(self.photo_viewer.btn_back)

        # 避让左侧 250px 看板，把文件名精准往右推到照片视口中轴上方！
        self.row2_viewer_layout.addSpacing(240)

        # 弹性区 1
        self.row2_viewer_layout.addStretch(1)

        # 中间：纯净文件名
        self.row2_viewer_layout.addWidget(self.photo_viewer.lbl_filename)

        # 弹性区 2：对称保持
        self.row2_viewer_layout.addStretch(1)

        # 右侧：方案 B 单排紧凑功能胶囊组 (高度 22px，水平无缝连排，彻底取代旧田字格)
        tool_capsule = QFrame()
        tool_capsule.setStyleSheet("QFrame { background-color: #18181c; border: 1px solid #2e2e36; border-radius: 3px; }")
        tool_capsule_l = QHBoxLayout(tool_capsule)
        tool_capsule_l.setContentsMargins(2, 2, 2, 2)
        tool_capsule_l.setSpacing(3)
        tool_capsule_l.addWidget(self.photo_viewer.btn_eye_zoom)
        tool_capsule_l.addWidget(self.photo_viewer.btn_af_toggle)
        tool_capsule_l.addWidget(self.photo_viewer.btn_compare)
        tool_capsule_l.addWidget(self.photo_viewer.btn_grid_toggle)
        tool_capsule_l.addWidget(self.photo_viewer.btn_crop_toggle)
        tool_capsule_l.addWidget(self.photo_viewer.combo_crop_ratio)
        tool_capsule_l.addWidget(self.photo_viewer.lbl_crop_live_ratio)
        tool_capsule_l.addWidget(self.photo_viewer.lbl_target_badge)
        self.row2_viewer_layout.addWidget(tool_capsule)

        self.page_stack.addWidget(self.photo_viewer)

        root_layout.addWidget(self.page_stack)

    def _load_initial_default_path(self):
        settings = QSettings("AnicropGeeks", "AnicropStudio")
        default_path = settings.value("default_album_path", "") or settings.value(self.settings_dir_key, "")
        if default_path and Path(default_path).exists() and Path(default_path).is_dir():
            self.load_folder(default_path, record_history=True)

    def _on_address_entered(self):
        text_path = self.edit_address.text().strip()
        p = Path(text_path)
        if p.exists() and p.is_dir():
            self.load_folder(str(p), record_history=True)
        else:
            QMessageBox.warning(self, "路径无效", f"无法访问指定的目录：\n{text_path}")
            self.edit_address.setText(self.current_dir)

    def _nav_go_back(self):
        if self.nav_index > 0:
            self.nav_index -= 1
            target = self.nav_history[self.nav_index]
            self.load_folder(target, record_history=False)
            self._update_nav_buttons()

    def _nav_go_forward(self):
        if self.nav_index < len(self.nav_history) - 1:
            self.nav_index += 1
            target = self.nav_history[self.nav_index]
            self.load_folder(target, record_history=False)
            self._update_nav_buttons()

    def _nav_go_up(self):
        if not self.current_dir:
            return
        cur_p = Path(self.current_dir)
        parent_p = cur_p.parent
        if parent_p and parent_p.exists() and parent_p != cur_p:
            self.load_folder(str(parent_p), record_history=True)

    def _update_nav_buttons(self):
        self.btn_nav_back.setEnabled(self.nav_index > 0)
        self.btn_nav_forward.setEnabled(self.nav_index < len(self.nav_history) - 1)

    def resizeEvent(self, event):
        super().resizeEvent(event)

    def _toggle_selection_mode(self):
        delegate = self.gallery.delegate
        delegate.is_selection_mode = not delegate.is_selection_mode
        if delegate.is_selection_mode:
            self.btn_select_mode.setText("✕ 取消选择")
            self.btn_select_mode.setStyleSheet("background-color: #7f1d1d; color: white; font-weight: bold; padding: 5px 12px;")
        else:
            self.btn_select_mode.setText("☑ 选择照片")
            self.btn_select_mode.setStyleSheet("QPushButton { background-color: #27272a; border: 1px solid #3f3f46; border-radius: 4px; padding: 5px 12px; font-weight: bold; } QPushButton:hover { background-color: #3f3f46; }")
            delegate.selected_set.clear()
            self.gallery.last_clicked_row = -1
            self.btn_enter_darkroom.setVisible(False)
        self.gallery.viewport().update()

    def _on_selection_updated(self):
        delegate = self.gallery.delegate
        sel_count = len(delegate.selected_set)
        if sel_count > 0:
            self.btn_enter_darkroom.setText(f"🧪 进入暗房 (已选 {sel_count} 张)  ➔")
            self.btn_enter_darkroom.setVisible(True)
        else:
            self.btn_enter_darkroom.setVisible(False)

    def _on_viewer_rating_requested(self, file_path: str, rating: int):
        """DPP 式打星: 写入 -> 回读校验 -> 三处联动(标题栏/卡片徽标/排序过滤)"""
        ok, msg = write_photo_rating(file_path, rating)
        if ok:
            # 穿透内存单趟缓存，确保回读直达物理磁盘
            _RAW_BUNDLE_CACHE.pop(file_path, None)
            if parse_photo_rating(file_path) == rating:
                self.ratings[file_path] = rating
                self.photo_viewer.update_rating_label(rating)
                self._rebuild_photo_list()
                self.gallery.viewport().update()
                return
        detail = msg if not ok else "写入后回读校验不一致，文件可能受写保护"
        QMessageBox.warning(self, "打星写入失败", f"{Path(file_path).name}\n\n{detail}")

    def _on_viewer_toggle_select(self, file_path: str):
        delegate = self.gallery.delegate
        if file_path == "__CHECK_STATE__":
            cur_p = self.photo_viewer.get_active_path()
            self.photo_viewer.update_mark_ui(cur_p in delegate.selected_set)
            return

        if file_path in delegate.selected_set:
            delegate.selected_set.remove(file_path)
            self.photo_viewer.update_mark_ui(False)
        else:
            delegate.selected_set.add(file_path)
            self.photo_viewer.update_mark_ui(True)
        self._on_selection_updated()
        self.gallery.viewport().update()

    def _on_enter_darkroom_clicked(self):
        delegate = self.gallery.delegate
        # 核心：从全格式总池（JPG+RAW）中提取全部勾选项，绝不遗漏跨模式挑选的照片！
        all_pool = list(self._all_jpgs) + [p for p in self._all_raws if p not in self._all_jpgs]
        selected_list = [p for p in all_pool if p in delegate.selected_set]
        if selected_list:
            self.sig_enter_darkroom.emit(selected_list, delegate.thumb_cache, self.current_dir)

    def _on_grid_slider_changed(self, val: int):
        self.gallery.set_base_card_width(val)
        pct = int((val / 220) * 100)
        self.lbl_slider_val.setText(f"缩略图: {pct}%")

    def _on_grid_zoom_step(self, step: int):
        new_val = max(130, min(420, self.slider_grid_size.value() + step))
        self.slider_grid_size.setValue(new_val)

    def apply_scroll_step(self, step: int):
        self.gallery.update_scroll_step(step)

    def _resolve_folder_cache_dir(self, source_folder: Path) -> Path:
        cache_root = get_app_cache_root()
        folder_name = source_folder.name.strip() or "Unnamed"
        candidate = cache_root / folder_name
        meta_file = candidate / ".source_meta"

        idx = 1
        while candidate.exists():
            if meta_file.exists():
                try:
                    if meta_file.read_text(encoding="utf-8").strip() == str(source_folder.resolve()):
                        return candidate
                except Exception:
                    pass
            candidate = cache_root / f"{folder_name}_{idx:03d}"
            meta_file = candidate / ".source_meta"
            idx += 1

        candidate.mkdir(parents=True, exist_ok=True)
        try:
            meta_file.write_text(str(source_folder.resolve()), encoding="utf-8")
        except Exception:
            pass
        return candidate

    def load_folder(self, folder_path: str, record_history: bool = True):
        p = Path(folder_path.strip())
        if not p.exists() or not p.is_dir():
            return

        self.current_dir = str(p)
        self.edit_address.setText(self.current_dir)
        QSettings("AnicropGeeks", "AnicropStudio").setValue(self.settings_dir_key, self.current_dir)

        if record_history:
            self.nav_history = self.nav_history[:self.nav_index + 1]
            self.nav_history.append(self.current_dir)
            self.nav_index = len(self.nav_history) - 1
            self._update_nav_buttons()

        self.active_cache_dir = self._resolve_folder_cache_dir(p)
        self.gallery.delegate.clear_cache()

        sub_dirs = []
        found_jpgs = []
        found_raws = []

        for f in p.iterdir():
            if f.is_dir() and not f.name.startswith("."):
                sub_dirs.append(str(f))
            elif f.is_file():
                ext = f.suffix.lower()
                if ext in VALID_IMAGE_EXTS:
                    found_jpgs.append(str(f))
                elif ext in RAW_IMAGE_EXTS:
                    found_raws.append(str(f))

        sub_dirs.sort()
        self._sub_dirs = sub_dirs
        self._all_jpgs = found_jpgs
        self._all_raws = found_raws
        self._rebuild_photo_list()

        if not self.photos:
            self.page_stack.setCurrentIndex(0)
            return

        self.warmup_total = len(self.photos)
        self.warmup_finished_count = 0
        self._warmup_done_fired = False
        _crashlog(f"[album] load_folder {p} 共{len(self.photos)}张")
        self.progress_loading.setValue(0)
        self.lbl_loading_detail.setText(f"预热进度: 0 / {self.warmup_total} (0%)")
        self.page_stack.setCurrentIndex(1)
        self.btn_open.setEnabled(False)

        thread_pool = QThreadPool.globalInstance()
        thread_pool.setMaxThreadCount(min(6, os.cpu_count() or 4))

        self._provisional_pending = set()
        self._upgrade_pool = QThreadPool(self)
        self._upgrade_pool.setMaxThreadCount(2)  # 升级梯队压低线程数，浏览时后台渐进补高清不抢资源

        # 【对象存续保险】QRunnable 默认 autoDelete，C++ 跑完就析构；Python 侧无引用又会被 GC——
        # 两边抢一个对象就是随机 access violation。挂引用 + 手动管生命周期，done 后才释放。
        self._live_thumb_tasks = set()
        for photo_path in self.photos:
            task = ThumbnailTask(photo_path, self.active_cache_dir, provisional_ok=True)
            task.setAutoDelete(False)
            self._live_thumb_tasks.add(task)
            task.signals.loaded.connect(self._on_single_thumb_cached)
            task.signals.task_done.connect(self._on_single_task_done)
            task.signals.meta_loaded.connect(self._on_rating_loaded)
            task.signals.provisional_used.connect(self._on_provisional_used)
            task.signals.task_done.connect(lambda p, t=task: self._live_thumb_tasks.discard(t))
            thread_pool.start(task)

    def _on_provisional_used(self, file_path: str):
        self._provisional_pending.add(file_path)

    def _start_thumb_upgrade_pass(self):
        """预热结束后后台渐进把 EXIF 小图升级为全质量 512px 缩略图（并落盘缓存，下次秒开）"""
        pending = list(self._provisional_pending)
        self._provisional_pending = set()
        _crashlog(f"[album] 升级梯队启动 {len(pending)} 张")
        for photo_path in pending:
            task = ThumbnailTask(photo_path, self.active_cache_dir, provisional_ok=False)
            task.setAutoDelete(False)  # 同预热：防 C++/GC 双重析构竞态
            self._live_thumb_tasks.add(task)
            task.signals.loaded.connect(self._on_single_thumb_cached)
            task.signals.task_done.connect(lambda p, t=task: self._live_thumb_tasks.discard(t))
            self._upgrade_pool.start(task)

    def _on_single_thumb_cached(self, file_path: str, qimage: QImage):
        # 信号跨线程传来 QImage（线程安全），此处是 GUI 线程，转 QPixmap 入缓存
        self.gallery.delegate.thumb_cache[file_path] = QPixmap.fromImage(qimage)
        if self.page_stack.currentIndex() == 0:
            # 重绘合并：升级梯队逐张到图，每张都全量重绘会把事件队列打爆（假死），80ms 合并成一帧
            if not getattr(self, "_thumb_repaint_timer", None):
                self._thumb_repaint_timer = QTimer(self)
                self._thumb_repaint_timer.setSingleShot(True)
                self._thumb_repaint_timer.setInterval(80)
                self._thumb_repaint_timer.timeout.connect(self.gallery.viewport().update)
            self._thumb_repaint_timer.start()

    def _on_single_task_done(self, file_path: str):
        self.warmup_finished_count += 1
        pct = int((min(self.warmup_finished_count, self.warmup_total) / self.warmup_total) * 100) if self.warmup_total > 0 else 100
        self.progress_loading.setValue(pct)
        self.lbl_loading_detail.setText(f"预热进度: {self.warmup_finished_count} / {self.warmup_total} ({pct}%)")

        # 幂等：完成收尾只允许执行一次（历史上 task_done 重复发射会把这里触发 N 遍，
        # 反复 reset 模型 + 重复启动升级梯队，用户点击时模型正在重建 → 闪退）
        if self.warmup_finished_count >= self.warmup_total and not getattr(self, "_warmup_done_fired", False):
            self._warmup_done_fired = True
            _crashlog(f"[album] 预热完成，待升级 {len(self._provisional_pending)} 张")
            self.page_stack.setCurrentIndex(0)
            self.btn_open.setEnabled(True)
            self._rebuild_photo_list()   # 打星全部就位，统一刷新排序与统计
            self.gallery.viewport().update()
            self._start_thumb_upgrade_pass()   # JPG 高清缩略图后台渐进补齐
            if self.photos:
                self.sig_selection_changed.emit(self.photos[0], 0, len(self.photos))

    def _on_sort_or_filter_changed(self):
        settings = QSettings("AnicropGeeks", "AnicropStudio")
        settings.setValue("album_format_mode", self.combo_format.currentData())
        settings.setValue("album_sort_mode", self.combo_sort.currentData())
        settings.setValue("album_filter_mode", self.combo_filter.currentData())
        self._rebuild_photo_list()
        self.gallery.viewport().update()

    def _rebuild_photo_list(self):
        """统一重建: 格式分流 -> 过滤 (打星) -> 排序 -> 刷新模型与统计"""
        fmt_mode = self.combo_format.currentData() if hasattr(self, 'combo_format') else "jpg"
        sort_mode = self.combo_sort.currentData() if hasattr(self, 'combo_sort') else "name"
        filter_mode = self.combo_filter.currentData() if hasattr(self, 'combo_filter') else "all"

        if fmt_mode == "raw":
            base_pool = list(self._all_raws)
        elif fmt_mode == "all":
            base_pool = list(self._all_jpgs) + [p for p in self._all_raws if p not in self._all_jpgs]
        else:
            base_pool = list(self._all_jpgs)

        photos = list(base_pool)
        if filter_mode == "rated":
            photos = [p for p in photos if self.ratings.get(p, 0) > 0]
        elif filter_mode == "unrated":
            photos = [p for p in photos if self.ratings.get(p, 0) <= 0]

# 文件夹与照片同步参与多维排序
        sub_dirs = list(self._sub_dirs)

        def _dir_mtime(d):
            try:
                return os.path.getmtime(d)
            except OSError:
                return 0.0

        if sort_mode == "mtime_desc":
            sub_dirs.sort(key=_dir_mtime, reverse=True)
            photos.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        elif sort_mode == "mtime_asc":
            sub_dirs.sort(key=_dir_mtime)
            photos.sort(key=lambda p: os.path.getmtime(p))
        elif sort_mode == "size_desc":
            # 文件夹按最新时间优先置顶，照片按物理字节大小倒序
            sub_dirs.sort(key=_dir_mtime, reverse=True)
            photos.sort(key=lambda p: os.path.getsize(p), reverse=True)
        elif sort_mode == "rating":
            # 文件夹按名称置顶，照片按星级高低倒序
            sub_dirs.sort(key=lambda d: Path(d).name.lower())
            photos.sort(key=lambda p: (-self.ratings.get(p, 0), Path(p).name.lower()))
        else:
            sub_dirs.sort(key=lambda d: Path(d).name.lower())
            photos.sort(key=lambda s: Path(s).name.lower())

        self.photos = photos
        all_items = [(d, True) for d in sub_dirs] + [(f, False) for f in photos]
        self.model.set_items(all_items)

        total_jpg = len(self._all_jpgs)
        total_raw = len(self._all_raws)
        rated_cnt = sum(1 for p in base_pool if self.ratings.get(p, 0) > 0)
        self.lbl_stats.setText(
            f"📁 {len(self._sub_dirs)} 文件夹 | 📷 {total_jpg} JPG | 🎞️ {total_raw} RAW | 展现 {len(photos)} 张 | ⭐ {rated_cnt} 打星"
        )

    def _on_rating_loaded(self, file_path: str, rating: int):
        self.ratings[file_path] = rating
        # 打星数据随缩略图管线流式到达，排序/统计在全量预热结束统一刷新，避免卡片抖动

    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择要检视的照片目录", self.current_dir or "")
        if folder:
            self.load_folder(folder, record_history=True)

    def _on_gallery_double_clicked(self, index: QModelIndex):
        if not index.isValid():
            return
        row = index.row()
        target_path, is_dir = self.model.get_item(row)

        if is_dir:
            self.load_folder(target_path, record_history=True)
        else:
            if target_path in self.photos:
                photo_idx = self.photos.index(target_path)
                is_checked = target_path in self.gallery.delegate.selected_set
                self.photo_viewer.load_context(
                    photo_idx, self.photos, 
                    thumb_cache=self.gallery.delegate.thumb_cache, 
                    is_marked=is_checked
                )
                self.row1_widget.setVisible(False)           # 隐藏第一行物理地址栏
                self.row2_gallery_box.setVisible(False)     # 隐藏画廊选择/排序/统计
                self.row2_viewer_box.setVisible(True)       # 显示大图检视器顶部栏
                self.page_stack.setCurrentIndex(2)
                self.photo_viewer.setFocus()
                self.sig_viewer_active.emit(True)

    def _on_viewer_back(self, return_index: int):
        self.row1_widget.setVisible(True)            # 恢复物理地址栏
        self.row2_gallery_box.setVisible(True)      # 恢复画廊工具栏
        self.row2_viewer_box.setVisible(False)      # 隐藏检视器顶部栏
        self.page_stack.setCurrentIndex(0)

        self.sig_viewer_active.emit(False)

        if 0 <= return_index < len(self.photos):
            photo_path = self.photos[return_index]
            for r in range(self.model.rowCount()):
                p, is_dir = self.model.get_item(r)
                if p == photo_path and not is_dir:
                    model_index = self.model.index(r, 0)
                    self.gallery.setCurrentIndex(model_index)
                    self.gallery.scrollTo(model_index, QAbstractItemView.PositionAtCenter)
                    break

    def _on_current_changed(self, current: QModelIndex, previous: QModelIndex):
        if current.isValid():
            row = current.row()
            path, is_dir = self.model.get_item(row)
            if not is_dir and path in self.photos:
                photo_idx = self.photos.index(path)
                self.sig_selection_changed.emit(path, photo_idx, len(self.photos))

# ==============================================================================
# P0.3 极客暗房工坊组件
# ==============================================================================
class DarkroomPageWidget(QWidget):
    sig_toggle_cinema_fullscreen = Signal()
    sig_selection_changed = Signal(str, int, int)
    sig_viewer_active = Signal(bool)
    sig_return_to_gallery = Signal()  # 【新增】返回画廊主相册物理信号

    def __init__(self, parent=None):
        super().__init__(parent)
        self.all_darkroom_photos = []
        self.visible_photos = []
        self.model = PhotoListModel()
        self.p02_source_dir = ""
        self.export_worker = None

        settings = QSettings("AnicropGeeks", "AnicropStudio")
        self.cull_mode = settings.value("darkroom_cull_mode", "reject")

        self._init_ui()

    def _init_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.stack = QStackedWidget()

        # 1. 大图检视器预先实例化（供顶部面板挂载控件）
        self.photo_viewer = SinglePhotoViewer(self, is_darkroom=True, cull_mode=self.cull_mode)
        self.photo_viewer.sig_back_to_gallery.connect(self._on_viewer_back)
        self.photo_viewer.sig_toggle_cinema_fullscreen.connect(self.sig_toggle_cinema_fullscreen)
        self.photo_viewer.sig_toggle_mark_requested.connect(self._on_viewer_toggle_hide)
        self.photo_viewer.sig_filter_mode_toggled.connect(self._on_viewer_filter_mode_toggled)
        self.photo_viewer.sig_photo_changed.connect(self.sig_selection_changed)
        self.photo_viewer.sig_rating_requested.connect(self._on_viewer_rating_requested)

        # 2. 暗房独立全宽顶部面板（支持画廊/大图双模式平滑切换）
        self.header_widget = QFrame()
        self.header_widget.setObjectName("DarkroomToolbar")
        self.header_widget.setStyleSheet("""
            QFrame#DarkroomToolbar {
                background-color: #141418;
                border-bottom: 1px solid #27272f;
                padding: 4px 6px;
            }
        """)
        h_vlayout = QVBoxLayout(self.header_widget)
        h_vlayout.setContentsMargins(6, 4, 6, 4)
        h_vlayout.setSpacing(0)

        # ---------------- 模式 A: 暗房画廊模式顶栏 ----------------
        self.gallery_bar = QWidget()
        b_layout = QHBoxLayout(self.gallery_bar)
        b_layout.setContentsMargins(0, 0, 0, 0)
        b_layout.setSpacing(8)

        # 【核心新增】一键返回画廊主相册物理按键，打通操作心流
        self.btn_back_to_album = QPushButton("← 返回画廊相册")
        self.btn_back_to_album.setStyleSheet("""
            QPushButton {
                background-color: #27272a;
                border: 1px solid #3f3f46;
                border-radius: 4px;
                padding: 4px 12px;
                font-weight: bold;
                color: #38bdf8;
            }
            QPushButton:hover {
                background-color: #3f3f46;
                color: #ffffff;
            }
        """)
        self.btn_back_to_album.setToolTip("返回漫展画廊初筛相册")
        self.btn_back_to_album.clicked.connect(self.sig_return_to_gallery.emit)
        b_layout.addWidget(self.btn_back_to_album)

        self.btn_remove_mode = QPushButton("🗑️ 移出照片")
        self.btn_remove_mode.setStyleSheet("font-weight: bold; padding: 4px 10px;")
        self.btn_remove_mode.clicked.connect(self._toggle_remove_mode)
        b_layout.addWidget(self.btn_remove_mode)

        self.btn_clear_darkroom = QPushButton("🧨 一键清空")
        self.btn_clear_darkroom.setStyleSheet("QPushButton { background-color: transparent; border: 1px solid #ef4444; border-radius: 4px; font-weight: bold; padding: 4px 10px; color: #ef4444; } QPushButton:hover { background-color: #7f1d1d; color: #ffffff; }")
        self.btn_clear_darkroom.setVisible(False)
        self.btn_clear_darkroom.clicked.connect(self._on_clear_darkroom)
        b_layout.addWidget(self.btn_clear_darkroom)

        cb_text = "显示未选中的照片 (变暗)" if self.cull_mode == "pick" else "显示隐藏的照片 (变暗)"
        self.cb_show_hidden = QCheckBox(cb_text)
        self.cb_show_hidden.setChecked(True)
        self.cb_show_hidden.setStyleSheet("font-weight: bold; color: #38bdf8;")
        self.cb_show_hidden.toggled.connect(self._on_toggle_show_hidden)
        b_layout.addWidget(self.cb_show_hidden)

        b_layout.addStretch()

        self.lbl_dark_stats = QLabel("暗房收录: 0 张")
        self.lbl_dark_stats.setStyleSheet("color: #a1a1aa; font-weight: bold;")
        b_layout.addWidget(self.lbl_dark_stats)

        h_vlayout.addWidget(self.gallery_bar)

        # ---------------- 模式 B: 暗房大图检视器专属顶栏 (36px 极客细轨) ----------------
        self.viewer_bar = QWidget()
        self.viewer_bar.setVisible(False)
        self.viewer_bar.setFixedHeight(36)
        self.viewer_layout = QHBoxLayout(self.viewer_bar)
        self.viewer_layout.setContentsMargins(0, 0, 6, 0)
        self.viewer_layout.setSpacing(0)

        # 左侧：正方形撑满高度的物理返回按键 "<"
        self.viewer_layout.addWidget(self.photo_viewer.btn_back)

        # 避让左侧 250px 看板，中轴绝对对称
        self.viewer_layout.addSpacing(240)
        self.viewer_layout.addStretch(1)
        self.viewer_layout.addWidget(self.photo_viewer.lbl_filename)
        self.viewer_layout.addStretch(1)

        # 右侧：满血接回胶囊工具组 (特写 / 对焦 / 对比 / 锁瞳徽标)
        tool_capsule = QFrame()
        tool_capsule.setStyleSheet("QFrame { background-color: #18181c; border: 1px solid #2e2e36; border-radius: 3px; }")
        tool_capsule_l = QHBoxLayout(tool_capsule)
        tool_capsule_l.setContentsMargins(2, 2, 2, 2)
        tool_capsule_l.setSpacing(3)
        tool_capsule_l.addWidget(self.photo_viewer.btn_eye_zoom)
        tool_capsule_l.addWidget(self.photo_viewer.btn_af_toggle)
        tool_capsule_l.addWidget(self.photo_viewer.btn_compare)
        tool_capsule_l.addWidget(self.photo_viewer.btn_grid_toggle)
        tool_capsule_l.addWidget(self.photo_viewer.btn_crop_toggle)
        tool_capsule_l.addWidget(self.photo_viewer.lbl_target_badge)
        self.viewer_layout.addWidget(tool_capsule)

        h_vlayout.addWidget(self.viewer_bar)

        # 3. 暗房网格画廊容器
        self.dark_container = QWidget()
        layout = QVBoxLayout(self.dark_container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.gallery = AdvancedGalleryView(self, is_darkroom=True, cull_mode=self.cull_mode)
        self.gallery.setModel(self.model)
        self.gallery.doubleClicked.connect(self._on_item_double_clicked)
        self.gallery.selectionModel().currentChanged.connect(self._on_current_changed)
        self.gallery.sig_zoom_requested.connect(self._on_grid_zoom_step)
        self.gallery.sig_selection_state_changed.connect(self._refresh_darkroom_view)
        self.gallery.sig_card_badge_clicked.connect(self._on_badge_clicked)
        layout.addWidget(self.gallery, stretch=1)

        bottom_bar = QFrame()
        bottom_bar.setStyleSheet("background-color: #18181b; border-radius: 6px; border: 1px solid #27272a;")
        bot_layout = QHBoxLayout(bottom_bar)
        bot_layout.setContentsMargins(12, 4, 12, 4)

        _mk = load_shortcuts()['mark']
        if self.cull_mode == "pick":
            hint_str = f"💡 提示：当前为【暗片挑好片】模式；按 {_mk} 点亮好片；未点亮的暗片不参与导出"
        else:
            hint_str = f"💡 提示：当前为【亮片踢废片】模式；按 {_mk} 压暗隐藏废片；被压暗的废片不参与导出"

        lbl_hint = QLabel(hint_str)
        lbl_hint.setStyleSheet("color: #71717a; font-size: 11px;")
        bot_layout.addWidget(lbl_hint)

        bot_layout.addStretch()

        self.btn_export = QPushButton("📤 一键导出交付 (自动生成三大件)")
        self.btn_export.setStyleSheet("QPushButton { background-color: #0284c7; color: white; font-weight: bold; font-size: 13px; border: none; border-radius: 6px; padding: 6px 20px; } QPushButton:hover { background-color: #0369a1; }")
        self.btn_export.clicked.connect(self._on_export_clicked)
        bot_layout.addWidget(self.btn_export)

        bot_layout.addStretch()

        self.lbl_slider_val = QLabel("缩略图: 100%", bottom_bar)
        self.lbl_slider_val.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 12px;")
        bot_layout.addWidget(self.lbl_slider_val)

        self.slider_grid_size = QSlider(Qt.Horizontal, bottom_bar)
        self.slider_grid_size.setRange(130, 420)
        self.slider_grid_size.setValue(220)
        self.slider_grid_size.setFixedWidth(140)
        self.slider_grid_size.valueChanged.connect(self._on_grid_slider_changed)
        bot_layout.addWidget(self.slider_grid_size)

        layout.addWidget(bottom_bar)
        self.stack.addWidget(self.dark_container)
        self.stack.addWidget(self.photo_viewer)

        root_layout.addWidget(self.stack)

    def load_from_gallery(self, selected_paths: list, thumb_cache: dict, source_dir: str):
        existing_set = set(self.all_darkroom_photos)
        new_photos = [p for p in selected_paths if p not in existing_set]
        self.all_darkroom_photos = list(self.all_darkroom_photos) + new_photos
        self.p02_source_dir = source_dir
        self.gallery.delegate.thumb_cache.update(thumb_cache)
        for p in new_photos:
            self.gallery.delegate.ratings[p] = parse_photo_rating(p)
            if self.cull_mode == "pick":
                self.gallery.delegate.hidden_set.add(p)

        self._refresh_darkroom_view()

    def _toggle_remove_mode(self):
        delegate = self.gallery.delegate
        delegate.is_remove_mode = not delegate.is_remove_mode
        if delegate.is_remove_mode:
            delegate.remove_set.clear()
            self.btn_remove_mode.setText("✕ 完成移出")
            self.btn_remove_mode.setStyleSheet("background-color: #7f1d1d; color: white; font-weight: bold; padding: 4px 10px;")
            self.btn_clear_darkroom.setVisible(len(self.all_darkroom_photos) > 0)
        else:
            self.btn_clear_darkroom.setVisible(False)
            self.btn_remove_mode.setText("🗑️ 移出照片")
            self.btn_remove_mode.setStyleSheet("font-weight: bold; padding: 4px 10px;")
            if delegate.remove_set:
                doomed = set(delegate.remove_set)
                self.all_darkroom_photos = [p for p in self.all_darkroom_photos if p not in doomed]
                delegate.hidden_set -= doomed
                delegate.remove_set.clear()
                self.gallery.last_clicked_row = -1
            self._refresh_darkroom_view()
        self.gallery.viewport().update()

    def _clear_darkroom_all(self):
        self.all_darkroom_photos.clear()
        self.gallery.delegate.hidden_set.clear()
        self.gallery.delegate.selected_set.clear()
        self.gallery.delegate.remove_set.clear()
        self.gallery.last_clicked_row = -1
        settings = QSettings("AnicropGeeks", "AnicropStudio")
        settings.remove("darkroom_photos")
        settings.remove("darkroom_hidden")
        settings.remove("darkroom_source_dir")
        try:
            restore_cache = get_app_cache_root() / "_darkroom_restore"
            if restore_cache.exists():
                shutil.rmtree(restore_cache, ignore_errors=True)
        except Exception:
            pass
        if self.gallery.delegate.is_remove_mode:
            self._toggle_remove_mode()
        self._refresh_darkroom_view()

    def _on_clear_darkroom(self):
        total = len(self.all_darkroom_photos)
        if total == 0:
            return
        ret = QMessageBox.question(
            self, "确认清空暗房",
            f"即将把暗房中全部 {total} 张照片一次性移出选片列表，并同步清空暗房缩略图缓存。\n\n仅移出暗房，不会删除硬盘上的原图文件。\n\n确认清空吗？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if ret != QMessageBox.Yes:
            return
        self._clear_darkroom_all()

    def showEvent(self, event):
        super().showEvent(event)
        if not getattr(self, "_exit_filter_installed", False):
            win = self.window()
            if win is not None and win is not self:
                win.installEventFilter(self)
                self._exit_filter_installed = True

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Close and getattr(self, "_exit_filter_installed", False):
            if obj is self.window() and self.all_darkroom_photos:
                total = len(self.all_darkroom_photos)
                box = QMessageBox(self.window())
                box.setWindowTitle("退出确认")
                box.setIcon(QMessageBox.Question)
                box.setText(f"暗房工坊里还有 {total} 张照片的选片现场。\n\n是否在退出前清空暗房？（不会删除硬盘上的原图文件）")
                btn_clear = box.addButton("清空并退出", QMessageBox.YesRole)
                btn_keep = box.addButton("保留现场退出", QMessageBox.NoRole)
                btn_cancel = box.addButton("取消", QMessageBox.RejectRole)
                box.setDefaultButton(btn_keep)
                box.exec()
                clicked = box.clickedButton()
                if clicked is btn_cancel:
                    event.ignore()
                    return True
                if clicked is btn_clear:
                    self._clear_darkroom_all()
        return super().eventFilter(obj, event)

    def _on_badge_clicked(self, file_path: str, action: str):
        if action == "remove":
            if file_path in self.all_darkroom_photos:
                self.all_darkroom_photos.remove(file_path)
            if file_path in self.gallery.delegate.hidden_set:
                self.gallery.delegate.hidden_set.remove(file_path)
            self._refresh_darkroom_view()

    def _on_toggle_show_hidden(self):
        self._refresh_darkroom_view()

    def _refresh_darkroom_view(self):
        delegate = self.gallery.delegate
        show_hidden = self.cb_show_hidden.isChecked()

        if show_hidden:
            self.visible_photos = list(self.all_darkroom_photos)
        else:
            self.visible_photos = [p for p in self.all_darkroom_photos if p not in delegate.hidden_set]

        self.model.set_photos(self.visible_photos)
        total_count = len(self.all_darkroom_photos)
        hidden_count = len(delegate.hidden_set)
        active_count = total_count - hidden_count
        jpg_cnt = sum(1 for p in self.all_darkroom_photos if not is_raw_file(p))
        raw_cnt = sum(1 for p in self.all_darkroom_photos if is_raw_file(p))
        fmt_detail = f"[{jpg_cnt} JPG, {raw_cnt} RAW]" if (jpg_cnt and raw_cnt) else (f"[{raw_cnt} RAW]" if raw_cnt else f"[{jpg_cnt} JPG]")

        if self.cull_mode == "pick":
            self.lbl_dark_stats.setText(f"暗房: {total_count} 张 {fmt_detail} (★已选好片 {active_count} | 待挑选 {hidden_count})")
        else:
            self.lbl_dark_stats.setText(f"暗房: {total_count} 张 {fmt_detail} (活跃保留 {active_count} | 剔除废片 {hidden_count})")

        if delegate.is_remove_mode:
            cnt = len(delegate.remove_set)
            self.btn_remove_mode.setText(f"✕ 完成移出 ({cnt})" if cnt > 0 else "✕ 完成移出")

        self.gallery.viewport().update()
        self._save_session()

    def _save_session(self):
        settings = QSettings("AnicropGeeks", "AnicropStudio")
        settings.setValue("darkroom_photos", self.all_darkroom_photos)
        settings.setValue("darkroom_hidden", list(self.gallery.delegate.hidden_set))
        settings.setValue("darkroom_source_dir", self.p02_source_dir)

    def restore_session(self):
        settings = QSettings("AnicropGeeks", "AnicropStudio")
        photos = settings.value("darkroom_photos", []) or []
        hidden = settings.value("darkroom_hidden", []) or []
        if isinstance(photos, str): photos = [photos]
        if isinstance(hidden, str): hidden = [hidden]
        photos = [p for p in photos if Path(p).exists()]
        if not photos:
            return

        self.all_darkroom_photos = photos
        self.p02_source_dir = settings.value("darkroom_source_dir", "")
        keep = set(photos)
        self.gallery.delegate.hidden_set = {p for p in hidden if p in keep}
        if not getattr(self, "_exit_filter_installed", False):
            win = self.window()
            if win is not None and win is not self:
                win.installEventFilter(self)
                self._exit_filter_installed = True
        self.gallery.delegate.ratings = {p: parse_photo_rating(p) for p in photos}
        self._refresh_darkroom_view()

        cache_dir = get_app_cache_root() / "_darkroom_restore"
        cache_dir.mkdir(parents=True, exist_ok=True)
        pool = QThreadPool.globalInstance()
        for p_str in photos:
            task = ThumbnailTask(p_str, cache_dir)
            task.setAutoDelete(False)
            self._restore_tasks = getattr(self, "_restore_tasks", set())
            self._restore_tasks.add(task)
            task.signals.loaded.connect(self._on_restore_thumb)
            task.signals.task_done.connect(lambda p, t=task: self._restore_tasks.discard(t))
            pool.start(task)

    def _on_restore_thumb(self, file_path: str, qimage: QImage):
        self.gallery.delegate.thumb_cache[file_path] = QPixmap.fromImage(qimage)
        if not getattr(self, "_restore_repaint_timer", None):
            self._restore_repaint_timer = QTimer(self)
            self._restore_repaint_timer.setSingleShot(True)
            self._restore_repaint_timer.setInterval(80)
            self._restore_repaint_timer.timeout.connect(self.gallery.viewport().update)
        self._restore_repaint_timer.start()

    def _on_grid_slider_changed(self, val: int):
        self.gallery.set_base_card_width(val)
        pct = int((val / 220) * 100)
        self.lbl_slider_val.setText(f"缩略图: {pct}%")

    def _on_grid_zoom_step(self, step: int):
        new_val = max(130, min(420, self.slider_grid_size.value() + step))
        self.slider_grid_size.setValue(new_val)

    def apply_scroll_step(self, step: int):
        self.gallery.update_scroll_step(step)

    def _on_current_changed(self, current: QModelIndex, previous: QModelIndex):
        if current.isValid():
            row = current.row()
            if 0 <= row < len(self.visible_photos):
                path = self.visible_photos[row]
                self.sig_selection_changed.emit(path, row, len(self.visible_photos))

    def _on_item_double_clicked(self, index: QModelIndex):
        if not index.isValid():
            return
        row = index.row()
        file_path = self.visible_photos[row]
        delegate = self.gallery.delegate

        show_unpicked_in_viewer = self.photo_viewer.cb_viewer_show_unpicked.isChecked()

        if show_unpicked_in_viewer:
            active_list = list(self.all_darkroom_photos)
            cur_idx = active_list.index(file_path) if file_path in active_list else 0
        else:
            active_list = [p for p in self.all_darkroom_photos if p not in delegate.hidden_set]
            if file_path in active_list:
                cur_idx = active_list.index(file_path)
            else:
                active_list = [file_path]
                cur_idx = 0

        is_marked = file_path in delegate.hidden_set
        self.photo_viewer.load_context(
            cur_idx, active_list, 
            thumb_cache=delegate.thumb_cache, 
            is_marked=is_marked
        )

        # 【核心切换】隐藏画廊工具栏，点亮大图检视顶栏细轨（含返回键与胶囊）
        self.gallery_bar.setVisible(False)
        self.viewer_bar.setVisible(True)
        self.stack.setCurrentIndex(1)
        self.sig_viewer_active.emit(True)
        self.photo_viewer.setFocus()

    def _on_viewer_filter_mode_toggled(self, show_all: bool):
        delegate = self.gallery.delegate
        if not (0 <= self.photo_viewer.current_index < len(self.photo_viewer.photo_list)):
            return
        cur_file = self.photo_viewer.photo_list[self.photo_viewer.current_index]

        if show_all:
            new_list = list(self.all_darkroom_photos)
        else:
            new_list = [p for p in self.all_darkroom_photos if p not in delegate.hidden_set]
            if not new_list:
                msg = "当前还没有挑选出任何好片！" if self.cull_mode == "pick" else "当前所有照片均已被标记为废片！"
                QMessageBox.information(self, "提示", msg)
                self.photo_viewer.cb_viewer_show_unpicked.blockSignals(True)
                self.photo_viewer.cb_viewer_show_unpicked.setChecked(True)
                self.photo_viewer.cb_viewer_show_unpicked.blockSignals(False)
                return

        new_idx = new_list.index(cur_file) if cur_file in new_list else 0
        is_marked = cur_file in delegate.hidden_set
        self.photo_viewer.load_context(new_idx, new_list, thumb_cache=delegate.thumb_cache, is_marked=is_marked)

    def _on_viewer_rating_requested(self, file_path: str, rating: int):
        ok, msg = write_photo_rating(file_path, rating)
        if ok:
            _RAW_BUNDLE_CACHE.pop(file_path, None)
            if parse_photo_rating(file_path) == rating:
                self.gallery.delegate.ratings[file_path] = rating
                self.photo_viewer.update_rating_label(rating)
                self.gallery.viewport().update()
                return
        detail = msg if not ok else "写入后回读校验不一致，文件可能受写保护"
        QMessageBox.warning(self, "打星写入失败", f"{Path(file_path).name}\n\n{detail}")

    def _on_viewer_toggle_hide(self, file_path: str):
        delegate = self.gallery.delegate
        if file_path == "__CHECK_STATE__":
            cur_p = self.photo_viewer.get_active_path()
            self.photo_viewer.update_mark_ui(cur_p in delegate.hidden_set)
            return

        if file_path in delegate.hidden_set:
            delegate.hidden_set.remove(file_path)
            self.photo_viewer.update_mark_ui(False)
        else:
            delegate.hidden_set.add(file_path)
            self.photo_viewer.update_mark_ui(True)

        self._refresh_darkroom_view()

    def _on_viewer_back(self, return_index: int):
        # 【核心切换】从大图退回时，恢复画廊工具栏，隐藏检视细轨
        self.gallery_bar.setVisible(True)
        self.viewer_bar.setVisible(False)
        self.stack.setCurrentIndex(0)
        self.sig_viewer_active.emit(False)
        self._refresh_darkroom_view()

    def _on_export_clicked(self):
        delegate = self.gallery.delegate
        unhidden_list = [p for p in self.all_darkroom_photos if p not in delegate.hidden_set]

        if not unhidden_list:
            msg = "当前暗房尚未挑选出任何好片，没有可导出的精选图！" if self.cull_mode == "pick" else "当前暗房所有照片均已被隐藏剔除，没有可导出的精选图！"
            QMessageBox.warning(self, "无有效照片", msg)
            return

        settings = QSettings("AnicropGeeks", "AnicropStudio")
        default_work_candidate = self.p02_source_dir or settings.value("last_work_root_dir", "") or settings.value("last_raw_source_dir", "")

        dlg = ExportDialog(self, default_work_dir=default_work_candidate)
        if dlg.exec() == QDialog.Accepted:
            dest_dir = dlg.final_dest_dir
            raw_src_dir = dlg.final_raw_src_dir
            smb_dest_dir = dlg.final_smb_dest_dir

            export_config = {
                "enable_all_jpg": settings.value("export_enable_all_jpg", True, type=bool),
                "name_all_jpg": str(settings.value("export_name_all_jpg", "jpg")),
                "enable_selected_jpg": settings.value("export_enable_selected_jpg", True, type=bool),
                "name_selected_jpg": str(settings.value("export_name_selected_jpg", "jpg2")),
                "enable_raw": settings.value("export_enable_raw", True, type=bool),
                "name_raw": str(settings.value("export_name_raw", "cr3")),
            }

            self.progress_dlg = QDialog(self)
            self.progress_dlg.setWindowTitle("正在执行暗房导出交付...")
            self.progress_dlg.resize(440, 120)
            p_layout = QVBoxLayout(self.progress_dlg)
            self.lbl_export_status = QLabel("正在初始化输出目录...")
            p_layout.addWidget(self.lbl_export_status)
            self.p_bar = QProgressBar()
            p_layout.addWidget(self.p_bar)
            self.progress_dlg.show()

            self.export_worker = DarkroomExportWorker(
                all_photos=self.all_darkroom_photos,
                unhidden_photos=unhidden_list,
                dest_dir=dest_dir,
                raw_src_dir=raw_src_dir,
                config=export_config,
                smb_dest_dir=smb_dest_dir
            )
            self.export_worker.sig_progress.connect(self._on_export_progress)
            self.export_worker.sig_finished.connect(self._on_export_finished)
            self.export_worker.start()

    def _on_export_progress(self, current: int, total: int, msg: str):
        pct = int((current / total) * 100) if total > 0 else 0
        self.p_bar.setValue(pct)
        self.lbl_export_status.setText(msg)

    def _on_export_finished(self, res: dict):
        self.progress_dlg.close()
        dlg = ExportSuccessDialog(res, self)
        dlg.exec()


class RawGalleryPageWidget(AlbumPageWidget):
    """🎞️ RAW 速览画廊：与漫展画廊完全同构（排序/打星/对比/暗房交付），
    唯一区别——只列 RAW 文件；缩略图与双击大图全部读取 RAW 内嵌的机内 JPEG 预览，
    不解析 RAW 本体，换来秒级浏览体验。
    """
    image_exts = RAW_IMAGE_EXTS
    settings_dir_key = "last_raw_dir"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.edit_address.setPlaceholderText("输入或粘贴 RAW 目录路径，回车秒级跳转 (仅识别 CR3/CR2/ARW/NEF/DNG/ORF/RW2)...")
        self.lbl_stats.setText("RAW 素材: 0 项")

