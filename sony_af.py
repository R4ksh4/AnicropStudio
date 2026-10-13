"""
索尼（Sony）AF 对焦点提取。

与佳能实现并列的独立解析路径，输出契约与 _canon_af_core 完全一致：
    [{"rect": QRectF(nx, ny, nw, nh), "in_focus": bool}, ...]
坐标为归一化(0~1) 左上原点，已按 EXIF orientation 旋正。

实测依据：ILCE-7M3（Sony ILCE-7M3 机身，ARW 内嵌预览 / 独立 JPEG 均验证通过）。
Sony MakerNote 的坑（务必保留注释，否则极易改错）：
  1. MakerNote 在 EXIF tag 0x927C，payload 头部是 b'SONY DSC\\x00'。
  2. 条目数在 payload offset 12，占2 字节；条目表从 offset 14 开始，每条 12 字节。
  3. 条目字节序是【小端】——但头部 bytes[8:12] 恰好是同一个小端整数，
     看起来像大端，极易误判。
  4. 【最容易错的一点】条目里的 value-offset 相对【整个 Exif/TIFF 块的起点】，
     而不是相对 MakerNote payload 起点。若误加/减 makernote 在 tiff 内的偏移，
     会读到形如 [125, 0, 215, 0] 的垃圾值，而不是真实坐标。
  5. Sony 用私有 tag 空间 0x2000–0xB054，不是 exiftool 文档里的 0x9xxx 区。
     对焦点在 0x2027（4×uint16：全图宽全图高 X Y），是唯一自带尺寸的 tag，最可靠。
"""

import struct

from PySide6.QtCore import QRectF

# EXIF / TIFF 标签
_EXIF_IFD_TAG = 0x8769
_MAKERNOTE_TAG = 0x927C
_ORIENTATION_TAG = 0x0112

# Sony MakerNote 私有标签（经 exiftool -v3 交叉验证）
_SONY_FOCUS_LOCATION = 0x2027      # uint16[4] -> full_w full_h x y
_SONY_FULL_IMAGE_SIZE = 0xB02B     # uint32[2] -> full_w full_h
_SONY_FOCUS_MODE = 0x201B# uint8[1]
_SONY_AF_POINTS_USED = 0x2020      # uint8[10] -> PDAF 位图

# TIFF 字段类型 -> (字节宽度, struct 字符)
_TYPE_INFO = {
    1: (1, 'B'), 2: (1, 'b'), 3: (2, 'H'), 4: (4, 'I'), 5: (8, None),
    6: (1, 'b'), 7: (1, 'B'), 8: (2, 'h'), 9: (4, 'i'), 10: (8, None),
    11: (4, 'f'), 12: (8, 'd'),
}

# 索尼典型对焦框边长占全图宽的比例；与佳能路径的 262.0 观感一致
_SONY_AF_BOX_RATIO = 0.039


def _tag_value_offset(tiff_data: bytes, endian: str, ep: int):
    """取一个 IFD 条目的值偏移：大值走值域，否则值就内联在 ep+8。"""
    t_field = struct.unpack(endian + 'H', tiff_data[ep + 2:ep + 4])[0]
    c_field = struct.unpack(endian + 'I', tiff_data[ep + 4:ep + 8])[0]
    if t_field == 7 and c_field > 4:
        return struct.unpack(endian + 'I', tiff_data[ep + 8:ep + 12])[0]
    return ep + 8


def _read_sony_makernote(tiff_data: bytes) -> bytes | None:
    """从 Exif/TIFF 块里定位并切出 Sony MakerNote payload。"""
    if len(tiff_data) < 8:
        return None
    endian = '<' if tiff_data[:2] == b'II' else ('>' if tiff_data[:2] == b'MM' else None)
    if endian is None:
        return None
    try:
        ifd0_off = struct.unpack(endian + 'I', tiff_data[4:8])[0]
        if ifd0_off + 2 > len(tiff_data):
            return None
        num_entries = struct.unpack(endian + 'H', tiff_data[ifd0_off:ifd0_off + 2])[0]

        makernote_off = None
        for i in range(num_entries):
            ep = ifd0_off + 2 + i * 12
            if ep + 12 > len(tiff_data):
                break
            tag = struct.unpack(endian + 'H', tiff_data[ep:ep + 2])[0]
            if tag == _MAKERNOTE_TAG:
                # 注意：0x927C 是 UNDEFINED(7)，大值时值域在 ep+8，且计数含尾部 NUL
                val_bytes = tiff_data[ep + 8:ep + 12]
                type_field = struct.unpack(endian + 'H', tiff_data[ep + 2:ep + 4])[0]
                count = struct.unpack(endian + 'I', tiff_data[ep + 4:ep + 8])[0]
                if type_field == 7 and count > 4:
                    makernote_off = struct.unpack(endian + 'I', val_bytes)[0]
                else:
                    makernote_off = ep + 8
                break
            elif tag == _EXIF_IFD_TAG:
                # MakerNote 常挂在 ExifIFD 而非 IFD0，继续记下 ExifIFD 位置
                exif_sub_off = struct.unpack(endian + 'I', tiff_data[ep + 8:ep + 12])[0]
                if makernote_off is None and exif_sub_off:
                    try:
                        n2 = struct.unpack(endian + 'H', tiff_data[exif_sub_off:exif_sub_off + 2])[0]
                        for j in range(n2):
                            q = exif_sub_off + 2 + j * 12
                            if q + 12 > len(tiff_data):
                                break
                            if struct.unpack(endian + 'H', tiff_data[q:q + 2])[0] == _MAKERNOTE_TAG:
                                t_field = struct.unpack(endian + 'H', tiff_data[q + 2:q + 4])[0]
                                c_field = struct.unpack(endian + 'I', tiff_data[q + 4:q + 8])[0]
                                if t_field == 7 and c_field > 4:
                                    makernote_off = struct.unpack(endian + 'I', tiff_data[q + 8:q + 12])[0]
                                else:
                                    makernote_off = q + 8
                                break
                    except Exception:
                        pass

        if not makernote_off:
            return None
        if makernote_off + 12 > len(tiff_data):
            return None
        mn = tiff_data[makernote_off:]
        if not mn.startswith(b'SONY'):
            return None
        return mn
    except Exception:
        return None


def _sony_tag_values(tiff_data: bytes, mn_off: int) -> dict:
    """解析 Sony MakerNote 条目表，返回 {tag: 原始值列表}。

    关键：value-offset 相对【tiff_data 起点】，所以这里以 tiff_data 为寻址基准，
    而不是以 MakerNote payload 为基准。
    """
    out = {}
    mn = tiff_data[mn_off:]
    try:
        count = struct.unpack_from('<H', mn, 12)[0]
    except Exception:
        return out
    if count == 0 or count > 512:
        return out

    for i in range(count):
        p = 14 + i * 12
        if p + 12 > len(mn):
            break
        try:
            tag, typ, cnt = struct.unpack_from('<HHI', mn, p)
        except Exception:
            break
        if cnt == 0:
            continue
        info = _TYPE_INFO.get(typ)
        if info is None:
            continue
        unit, ch = info
        size = unit * cnt
        raw = mn[p + 8:p + 12]
        try:
            if size <= 4:
                if ch is None:
                    continue
                out[tag] = [struct.unpack_from('<' + ch + str(cnt), raw, 0)[0]]
                continue
            # 基准是整个 tiff_data —— 见模块 docstring 第 4 条
            vo = struct.unpack_from('<I', raw, 0)[0]
            if vo < 0 or vo + size > len(tiff_data):
                continue
            if typ == 2:
                out[tag] = tiff_data[vo:vo + cnt].split(b'\x00')[0].decode('ascii', 'replace').strip()
            elif typ == 7 or ch is None:
                out[tag] = tiff_data[vo:vo + min(cnt, 32)]
            else:
                out[tag] = list(struct.unpack_from('<' + str(cnt) + ch, tiff_data, vo))
        except Exception:
            continue
    return out


def _rotate_norm(nx: float, ny: float, nw: float, nh: float, orientation: int):
    """把归一化坐标按 EXIF orientation 旋正，返回 (nx, ny, nw, nh)。

    与 _canon_af_core 里的旋转约定保持一致（norm_cx/norm_cy 为中心点）。
    """
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


def _sony_af_core(tiff_data: bytes, endian: str = '<', orientation: int = 1) -> list:
    """Sony MakerNote 解析，输出与 _canon_af_core 同构的框列表。"""
    try:
        head = tiff_data[:2]
        if head == b'II':
            endian = '<'
        elif head == b'MM':
            endian = '>'
        else:
            return []
        ifd0_off = struct.unpack(endian + 'I', tiff_data[4:8])[0]
        if ifd0_off + 2 > len(tiff_data):
            return []
        num_entries = struct.unpack(endian + 'H', tiff_data[ifd0_off:ifd0_off + 2])[0]

        mn_off = None
        exif_off = None
        for i in range(num_entries):
            ep = ifd0_off + 2 + i * 12
            if ep + 12 > len(tiff_data):
                break
            tag = struct.unpack(endian + 'H', tiff_data[ep:ep + 2])[0]
            if tag == _MAKERNOTE_TAG:
                mn_off = _tag_value_offset(tiff_data, endian, ep)
                break
            elif tag == _EXIF_IFD_TAG:
                exif_off = struct.unpack(endian + 'I', tiff_data[ep + 8:ep + 12])[0]

        # 【关键】索尼的 0x927C 挂在 ExifIFD(0x8769) 下，不在 IFD0 —— 必须再扫一级，
        # 只扫 IFD0 会得到空结果（索尼与佳能把 MakerNote 挂的位置不同）。
        if mn_off is None and exif_off:
            try:
                if exif_off + 2 <= len(tiff_data):
                    n2 = struct.unpack(endian + 'H', tiff_data[exif_off:exif_off + 2])[0]
                    for j in range(n2):
                        q = exif_off + 2 + j * 12
                        if q + 12 > len(tiff_data):
                            break
                        if struct.unpack(endian + 'H', tiff_data[q:q + 2])[0] == _MAKERNOTE_TAG:
                            mn_off = _tag_value_offset(tiff_data, endian, q)
                            break
            except Exception:
                pass

        if not mn_off or mn_off + 12 > len(tiff_data):
            return []
        if not tiff_data[mn_off:].startswith(b'SONY'):
            return []

        tags = _sony_tag_values(tiff_data, mn_off)
        if not tags:
            return []

        full_w = full_h = 0
        fx = fy = None

        # 优先 0x2027 FocusLocation：自带全图尺寸与坐标，一个 tag 搞定
        fl = tags.get(_SONY_FOCUS_LOCATION)
        if fl and len(fl) == 4 and all(0 < v < 30000 for v in fl[:2]):
            full_w, full_h, fx, fy = fl[0], fl[1], fl[2], fl[3]

        if not full_w or not full_h:
            fis = tags.get(_SONY_FULL_IMAGE_SIZE)
            if fis and len(fis) == 2 and all(0 < v < 30000 for v in fis):
                full_w, full_h = fis[0], fis[1]

        if not full_w or not full_h:
            return []

        boxes = []
        # 0x2027 全为 0 表示未记录对焦点（PDAF 场景常见），此时不画框
        if fx is not None and fx > 0 and fy > 0 and fx < full_w and fy < full_h:
            nx = fx / float(full_w)
            ny = fy / float(full_h)
            nw = _SONY_AF_BOX_RATIO
            nh = _SONY_AF_BOX_RATIO
            rcx, rcy, rw, rh = _rotate_norm(nx, ny, nw, nh, orientation)
            # 【契约对齐】绘制层按 QRectF 的左上角 +宽高用（见paintEvent里
            # bx = draw_x + norm_r.x() * draw_w），这里必须给左上角而非中心点，
            # 否则框会整体偏移半个框宽。佳能的 _canon_af_core 同样是先中心再减半。
            rx = max(0.0, min(rcx - rw / 2.0, 1.0 - rw))
            ry = max(0.0, min(rcy - rh / 2.0, 1.0 - rh))
            boxes.append({
                "rect": QRectF(rx, ry, rw, rh),
                "in_focus": True,
            })

        # PDAF 点阵：0x2020 为 10 字节标志位，标出参与合成的相位对焦点
        apu = tags.get(_SONY_AF_POINTS_USED)
        if apu and isinstance(apu, (bytes, bytearray)) and len(apu) >= 10:
            if any(apu[:10]):
                # 本批 ILCE-7M3 样本该值恒为 0，此分支暂不产出框，
                # 保留是为了后续机型接入时不改结构。
                pass
        return boxes
    except Exception:
        return []


def _sony_af_from_jpeg_bytes(data: bytes, orientation: int = 1) -> list:
    """从 JPEG 字节流的 APP1/EXIF 里解索尼 AF 框。"""
    try:
        if len(data) < 4 or data[:2] != b'\xFF\xD8':
            return []
        idx = 2
        n = len(data)
        while idx < n - 4:
            if data[idx] != 0xFF:
                break
            marker = data[idx + 1]
            if marker in (0xD9, 0xDA):
                break
            if marker == 0xD8 or marker == 0x01 or 0xD0 <= marker <= 0xD7:
                idx += 2
                continue
            length = int.from_bytes(data[idx + 2:idx + 4], 'big')
            if marker == 0xE1:
                body = data[idx + 4:idx + 2 + length]
                if body.startswith(b'Exif\x00\x00'):
                    tiff_data = body[6:]
                    if tiff_data[:2] in (b'II', b'MM'):
                        boxes = _sony_af_core(tiff_data, '<', orientation)
                        if boxes:
                            return boxes
            idx += 2 + length
        return []
    except Exception:
        return []


def extract_sony_af_points(file_path: str, orientation: int = 1) -> list:
    """索尼 AF 框提取（仅 JPEG/内嵌预览 JPEG 路径）。"""
    try:
        with open(file_path, 'rb') as f:
            data = f.read(250000)
        if len(data) >= 4 and data[:2] == b'\xFF\xD8':
            return _sony_af_from_jpeg_bytes(data, orientation)
        return []
    except Exception:
        return []
