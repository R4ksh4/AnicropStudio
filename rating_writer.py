# -*- coding: utf-8 -*-
"""
XMP 打星写入引擎 (xmp:Rating 0~5)
兼容: JPEG (APP1 XMP 段) / 佳能 CR3·CR2 (BMFF uuid XMP 盒)
安全铁律:
  1. 只改写 XMP 元数据区，绝不触碰图像像素数据
  2. 临时文件 + os.replace 原子替换，中途断电不产生残尸
  3. 保留原始 mtime，缩略图缓存不失效
"""
import os
import re
import struct
from pathlib import Path

XMP_PACKET_HEADER = b'http://ns.adobe.com/xap/1.0/\x00'
XMP_UUID = bytes.fromhex('BE7ACFCB97A942E89C71999491E3AFAC')  # 标准 XMP uuid

_ATTR_RE = re.compile(rb'(xmp:Rating\s*=\s*")(-?\d+)(")')
_ELEM_RE = re.compile(rb'(<xmp:Rating>)(-?\d+)(</xmp:Rating>)')


def _minimal_packet(rating: int) -> bytes:
    """无 XMP 包时的最小合规包 (DPP / Lightroom 均可识别)"""
    return (
        '<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="AnicropStudio">\n'
        ' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
        '  <rdf:Description rdf:about="" xmlns:xmp="http://ns.adobe.com/xap/1.0/" '
        f'xmp:Rating="{rating}"/>\n'
        ' </rdf:RDF>\n'
        '</x:xmpmeta>\n'
        '<?xpacket end="w"?>'
    ).encode('utf-8')


def _patch_packet(packet: bytes, rating: int) -> bytes | None:
    """在既有 XMP 包内改写星级。返回 None 表示包结构异常需整体替换。"""
    rating_b = str(rating).encode('ascii')
    if _ATTR_RE.search(packet):
        return _ATTR_RE.sub(rb'\g<1>' + rating_b + rb'\g<3>', packet, count=1)
    if _ELEM_RE.search(packet):
        return _ELEM_RE.sub(rb'\g<1>' + rating_b + rb'\g<3>', packet, count=1)

    # 包里没有 Rating 字段: 向第一个 rdf:Description 注入属性 (需已声明 xmlns:xmp)
    if b'xmlns:xmp' not in packet:
        return None
    m = re.search(rb'<rdf:Description\b', packet)
    if not m:
        return None
    tag_end = packet.find(b'>', m.end())
    if tag_end < 0:
        return None
    insert_at = tag_end - 1 if packet[tag_end - 1:tag_end] == b'/' else tag_end
    injection = f' xmp:Rating="{rating}"'.encode('ascii')
    if insert_at != tag_end:
        # 自闭合标签: 插到 "/" 之前
        return packet[:insert_at] + injection + packet[insert_at:]
    return packet[:tag_end] + injection + packet[tag_end:]


def _atomic_write(path: Path, data: bytes) -> tuple[bool, str]:
    st = path.stat()
    tmp = path.parent / (path.name + '.anicrop_tmp')
    try:
        with open(tmp, 'wb') as f:
            f.write(data)
        os.utime(tmp, (st.st_atime, st.st_mtime))
        os.replace(tmp, path)
        return True, "OK"
    except Exception as e:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        return False, str(e)


def _write_jpg(path: Path, rating: int) -> tuple[bool, str]:
    data = path.read_bytes()
    if len(data) < 4 or data[:2] != b'\xff\xd8':
        return False, "不是有效的 JPEG 文件"

    idx = 2
    insert_pos = 2  # 无 XMP 时的插入点: 所有头部 APPn 段之后
    while idx < len(data) - 4:
        if data[idx] != 0xFF:
            break
        marker = data[idx + 1]
        if marker in (0xD9, 0xDA):
            break
        length = int.from_bytes(data[idx + 2:idx + 4], 'big')
        seg_start, seg_end = idx, idx + 2 + length
        body = data[idx + 4: seg_end]

        if marker == 0xE1 and body.startswith(XMP_PACKET_HEADER):
            packet = body[len(XMP_PACKET_HEADER):]
            new_packet = _patch_packet(packet, rating)
            if new_packet is None:
                new_packet = _minimal_packet(rating)
            new_body = XMP_PACKET_HEADER + new_packet
            if len(new_body) + 2 > 65533:
                return False, "XMP 段超过 JPEG 长度上限"
            new_seg = b'\xff\xe1' + (len(new_body) + 2).to_bytes(2, 'big') + new_body
            return _atomic_write(path, data[:seg_start] + new_seg + data[seg_end:])

        idx = seg_end
        if marker in (0xE0, 0xE1, 0xE2, 0xEE):
            insert_pos = seg_end

    # 没有 XMP 段: 新建一个 APP1 插入头部
    body = XMP_PACKET_HEADER + _minimal_packet(rating)
    seg = b'\xff\xe1' + (len(body) + 2).to_bytes(2, 'big') + body
    return _atomic_write(path, data[:insert_pos] + seg + data[insert_pos:])


def _walk_top_boxes(data: bytes):
    """遍历 BMFF (CR3/CR2 新一代容器) 顶层 box，产出 (偏移, 头长, 总长度, 类型)"""
    off = 0
    while off + 8 <= len(data):
        size = int.from_bytes(data[off:off + 4], 'big')
        typ = data[off + 4:off + 8]
        hdr = 8
        if size == 1:
            if off + 16 > len(data):
                break
            size = int.from_bytes(data[off + 8:off + 16], 'big')
            hdr = 16
        elif size == 0:
            size = len(data) - off
        if size < hdr or off + size > len(data):
            break
        yield off, hdr, size, typ
        off += size


def _write_cr3(path: Path, rating: int) -> tuple[bool, str]:
    data = path.read_bytes()
    if len(data) < 16 or data[4:8] not in (b'ftyp', b'wide', b'moov', b'mdat', b'uuid', b'free'):
        return False, "不是有效的 CR3/CR2 容器结构"

    for off, hdr, size, typ in _walk_top_boxes(data):
        if typ == b'uuid' and data[off + hdr: off + hdr + 16] == XMP_UUID:
            packet = data[off + hdr + 16: off + size]
            new_packet = _patch_packet(packet, rating)
            if new_packet is None:
                new_packet = _minimal_packet(rating)
            new_size = hdr + 16 + len(new_packet)
            if hdr == 8:
                new_box = new_size.to_bytes(4, 'big') + b'uuid' + XMP_UUID + new_packet
            else:
                new_box = (1).to_bytes(4, 'big') + b'uuid' + new_size.to_bytes(8, 'big') + XMP_UUID + new_packet
            return _atomic_write(path, data[:off] + new_box + data[off + size:])

    # 无 XMP 盒: 末尾追加一个顶层 uuid box (BMFF 顶层盒顺序无关，读取器可正常发现)
    packet = _minimal_packet(rating)
    box = (8 + 16 + len(packet)).to_bytes(4, 'big') + b'uuid' + XMP_UUID + packet
    return _atomic_write(path, data + box)

def _write_arw(path: Path, rating: int) -> tuple[bool, str]:
    """
    索尼 ARW 原地安全原子改写引擎:
    1. 优先原地覆写 Sony MakerNote Tag 0x2002 (4字节内联 uint32，机身/LR双互认)
    2. 若内嵌 XMP 包，执行 1:1 绝对等长单字符改写，保持 TIFF 偏移链表绝对守恒
    3. 全程文件总字节数 1 字节不增不减，绝对不触碰 RAW 像素
    """
    data = bytearray(path.read_bytes())
    if len(data) < 16 or data[:4] not in (b'II*\x00', b'MM\x00*'):
        return False, "不是有效的 ARW/TIFF 容器结构"

    modified = False
    scan_limit = min(len(data), 4 * 1024 * 1024)

    # 1. 扫描 Sony MakerNote Tag 0x2002 原地覆写
    pos = 0
    while True:
        idx = data.find(b'SONY DSC', pos, scan_limit)
        if idx == -1:
            break
        if idx + 14 <= len(data):
            cnt = struct.unpack_from('<H', data, idx + 12)[0]
            if 0 < cnt <= 512:
                for i in range(cnt):
                    p = idx + 14 + i * 12
                    if p + 12 > len(data):
                        break
                    tag = struct.unpack_from('<H', data, p)[0]
                    if tag == 0x2002:
                        data[p + 8 : p + 12] = struct.pack('<I', rating)
                        modified = True
                        break
                if modified:
                    break
        pos = idx + 8

    # 2. 若 ARW 内部带有内嵌 XMP，同步执行 1:1 绝对等长字符替换
    xmp_idx = data.find(b'<x:xmpmeta', 0, scan_limit)
    if xmp_idx >= 0:
        xmp_end = data.find(b'</x:xmpmeta>', xmp_idx, xmp_idx + 131072)
        if xmp_end > 0:
            xmp_chunk = bytes(data[xmp_idx : xmp_end + 12])
            rating_b = str(rating).encode('ascii')
            m_attr = re.search(rb'(xmp:Rating\s*=\s*")(\d)(")', xmp_chunk)
            if m_attr:
                offset = xmp_idx + m_attr.start(2)
                data[offset : offset + 1] = rating_b
                modified = True
            else:
                m_elem = re.search(rb'(<xmp:Rating>)(\d)(</xmp:Rating>)', xmp_chunk)
                if m_elem:
                    offset = xmp_idx + m_elem.start(2)
                    data[offset : offset + 1] = rating_b
                    modified = True

    if not modified:
        return False, "ARW 结构中未检索到可原地安全改写的打星段 (0x2002/XMP)"

    return _atomic_write(path, bytes(data))


def write_photo_rating(file_path: str, rating: int) -> tuple[bool, str]:
    """
    统一入口: 给照片写入机内兼容星级 (0=清除, 1~5)。
    返回 (是否成功, 信息)。不支持的格式明确拒绝，绝不硬写。
    """
    try:
        rating = int(rating)
    except (TypeError, ValueError):
        return False, "星级必须为整数"
    if not 0 <= rating <= 5:
        return False, "星级须在 0~5 之间"

    p = Path(file_path)
    if not p.exists():
        return False, "文件不存在"

    ext = p.suffix.lower()
    try:
        if ext in ('.jpg', '.jpeg'):
            return _write_jpg(p, rating)
        if ext in ('.cr3', '.cr2'):
            return _write_cr3(p, rating)
        if ext in ('.arw',):
            return _write_arw(p, rating)
        return False, f"暂不支持向 {ext or '该格式'} 写入打星 (当前支持 JPG / CR3 / CR2 / ARW)"
    except PermissionError:
        return False, "文件被占用或没有写入权限"
    except Exception as e:
        return False, f"写入异常: {e}"
