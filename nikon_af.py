# -*- coding: utf-8 -*-
"""
nikon_af.py - 尼康（Nikon）AF 对焦点提取引擎
全面适配：尼康 Z 系列微单 (Z9/Z8/Z6/Z7/Zf/Z50/Z5 等) 与 D 系列单反 (D850/D750/D500 等)。
协议依据：Nikon MakerNote Tag 0x00B7 (AFInfo2)。
输出契约：[{"rect": QRectF(rx, ry, rw, rh), "in_focus": True}]
坐标系：归一化 (0~1) 左上原点，已根据 EXIF Orientation 旋正。
"""

import struct
from PySide6.QtCore import QRectF

_MAKERNOTE_TAG = 0x927C
_EXIF_IFD_TAG = 0x8769
_NIKON_AFINFO2_TAG = 0x00B7

# 默认对焦框占比（若机身只记录点坐标没记录长宽时使用）
_NIKON_DEFAULT_BOX_RATIO = 0.038


def _rotate_norm(nx: float, ny: float, nw: float, nh: float, orientation: int):
    """根据 EXIF 方向旋正归一化中心点 (nx, ny) 与尺寸 (nw, nh)"""
    if orientation == 6:
        return 1.0 - ny, nx, nh, nw
    if orientation == 8:
        return ny, 1.0 - nx, nh, nw
    if orientation == 3:
        return 1.0 - nx, 1.0 - ny, nw, nh
    if orientation == 2:
        return 1.0 - nx, ny, nw, nh
    if orientation == 4:
        return nx, 1.0 - ny, nw, nh
    return nx, ny, nw, nh


def _parse_afinfo2_payload(payload: bytes, orientation: int = 1) -> list:
    if len(payload) < 28:
        return []

    af_w, af_h, x_pos, y_pos, box_w, box_h = 0, 0, 0, 0, 0, 0
    ver = payload[:4]

    # 1. 尝试各代 Expeed 引擎版本固定偏移
    candidate_offsets = []
    if ver in (b"0400", b"0401"):          # Expeed 7: Z9, Z8, Zf, Z6 III 等
        candidate_offsets.append(62)
    elif ver in (b"0300", b"0301"):        # Expeed 6: Z6, Z7, Z5, Z50, Zfc, D6, D780
        candidate_offsets.append(70)
    elif ver == b"0101":                   # D5, D500, D850, D7500
        candidate_offsets.extend([42, 70])
    elif ver in (b"0100", b"0200"):        # 早期 LiveView DSLRs / Nikon 1
        candidate_offsets.append(16)

    # 2. 匹配固定偏移并进行几何自校验
    matched = False
    for off in candidate_offsets:
        if off + 12 <= len(payload):
            w_img, h_img, x, y, bw, bh = struct.unpack_from('<6H', payload, off)
            if 800 < w_img < 16000 and 800 < h_img < 16000 and 0 < x < w_img and 0 < y < h_img:
                af_w, af_h, x_pos, y_pos, box_w, box_h = w_img, h_img, x, y, bw, bh
                matched = True
                break

    # 3. 容错扫描：若版本未知或偏移偏移，直接在有效载荷内检索合法的连续 6 个 uint16 特征
    if not matched:
        for p in range(16, min(len(payload) - 12, 128), 2):
            w_img, h_img, x, y, bw, bh = struct.unpack_from('<6H', payload, p)
            if 800 < w_img < 16000 and 800 < h_img < 16000 and 0 < x < w_img and 0 < y < h_img:
                if 0 <= bw < (w_img / 2.0) and 0 <= bh < (h_img / 2.0):
                    af_w, af_h, x_pos, y_pos, box_w, box_h = w_img, h_img, x, y, bw, bh
                    matched = True
                    break

    if not matched or af_w <= 0 or af_h <= 0:
        return []

    nx = x_pos / float(af_w)
    ny = y_pos / float(af_h)
    nw = (box_w / float(af_w)) if box_w > 0 else _NIKON_DEFAULT_BOX_RATIO
    nh = (box_h / float(af_h)) if box_h > 0 else _NIKON_DEFAULT_BOX_RATIO

    rcx, rcy, rw, rh = _rotate_norm(nx, ny, nw, nh, orientation)
    rx = max(0.0, min(rcx - rw / 2.0, 1.0 - rw))
    ry = max(0.0, min(rcy - rh / 2.0, 1.0 - rh))

    return [{
        "rect": QRectF(rx, ry, rw, rh),
        "in_focus": True,
    }]


def _nikon_af_core(tiff_data: bytes, endian: str = '<', orientation: int = 1) -> list:
    """从 TIFF 格式块（NEF 主干或 EXIF 块）中提取尼康 AF 框"""
    if len(tiff_data) < 16:
        return []

    # 阶段 1: 扫描 MakerNote 块定位
    mn_payload = None

    # 路径 A: 标准 IFD 寻找 0x927C
    try:
        if tiff_data[:2] in (b'II', b'MM'):
            endian = '<' if tiff_data[:2] == b'II' else '>'
            ifd0_off = struct.unpack(endian + 'I', tiff_data[4:8])[0]
            cur_ifds = [ifd0_off]
            
            # 取 ExifIFD
            num_e = struct.unpack(endian + 'H', tiff_data[ifd0_off:ifd0_off + 2])[0]
            for i in range(min(num_e, 100)):
                ep = ifd0_off + 2 + i * 12
                if ep + 12 > len(tiff_data):
                    break
                tag = struct.unpack(endian + 'H', tiff_data[ep:ep + 2])[0]
                if tag == _EXIF_IFD_TAG:
                    cur_ifds.append(struct.unpack(endian + 'I', tiff_data[ep + 8:ep + 12])[0])
                    break

            for ifd in cur_ifds:
                if ifd + 2 > len(tiff_data):
                    continue
                cnt = struct.unpack(endian + 'H', tiff_data[ifd:ifd + 2])[0]
                for i in range(min(cnt, 120)):
                    ep = ifd + 2 + i * 12
                    if ep + 12 > len(tiff_data):
                        break
                    tag = struct.unpack(endian + 'H', tiff_data[ep:ep + 2])[0]
                    if tag == _MAKERNOTE_TAG:
                        val_off = struct.unpack(endian + 'I', tiff_data[ep + 8:ep + 12])[0]
                        if val_off < len(tiff_data):
                            mn_payload = tiff_data[val_off:]
                        break
                if mn_payload:
                    break
    except Exception:
        pass

    # 路径 B: 硬件特征极速兜底 (搜 b'Nikon\\x00\\x02')
    if not mn_payload:
        pos = tiff_data.find(b'Nikon\x00\x02')
        if pos != -1:
            mn_payload = tiff_data[pos:]

    if not mn_payload or len(mn_payload) < 18:
        return []

    # 阶段 2: 尼康 Type 3 MakerNote 带有 10 字节头，之后是专属嵌入式 TIFF 头
    try:
        mn_tiff = mn_payload[10:] if mn_payload.startswith(b'Nikon\x00\x02') else mn_payload
        if mn_tiff[:2] not in (b'II', b'MM'):
            return []
        mn_endian = '<' if mn_tiff[:2] == b'II' else '>'
        ifd_off = struct.unpack(mn_endian + 'I', mn_tiff[4:8])[0]
        if ifd_off + 2 > len(mn_tiff):
            return []

        num_entries = struct.unpack(mn_endian + 'H', mn_tiff[ifd_off:ifd_off + 2])[0]
        for i in range(min(num_entries, 200)):
            ep = ifd_off + 2 + i * 12
            if ep + 12 > len(mn_tiff):
                break
            tag, ftype, count = struct.unpack(mn_endian + 'HHI', mn_tiff[ep:ep + 8])
            if tag == _NIKON_AFINFO2_TAG:
                val_raw = struct.unpack(mn_endian + 'I', mn_tiff[ep + 8:ep + 12])[0]
                total_bytes = count
                afinfo2_bytes = mn_tiff[val_raw:val_raw + total_bytes]
                return _parse_afinfo2_payload(afinfo2_bytes, orientation)
    except Exception:
        pass

    return []