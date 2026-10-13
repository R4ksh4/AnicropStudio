"""
索尼 AF 集成验证脚本（自包含，无需额外依赖文件）。

用法：
    python verify_sony_integration.py                # 抽样验证 + 渲染对比图
    python verify_sony_integration.py --full         # 全量4012 张比对

检查两件事：
  1. extract_af_points() 对索尼照片能稳定产出对焦点框；
  2. 框中心与「独立解析算出的坐标」一致（用项目 paintEvent 的换算口径复刻）。
"""
import glob
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import album

SONY_FOCUS_LOCATION = 0x2027


# ---------- 独立实现一遍 Sony 解析（不复用 sony_af.py，避免自证） ----------
def _tiff_of(data: bytes):
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


def _orientation(tiff):
    en = '<' if tiff[:2] == b'II' else '>'
    try:
        ifd0 = struct.unpack_from(en + 'I', tiff, 4)[0]
        cnt = struct.unpack_from(en + 'H', tiff, ifd0)[0]
        for i in range(cnt):
            p = ifd0 + 2 + i * 12
            if struct.unpack_from(en + 'H', tiff, p)[0] == 0x0112:
                return struct.unpack_from(en + 'H', tiff, p + 8)[0]
    except Exception:
        pass
    return 1


def _sony_focus_independent(tiff):
    """独立解析：返回 (fw, fh, x, y) 或 None。"""
    en = '<' if tiff[:2] == b'II' else '>'
    try:
        ifd0 = struct.unpack_from(en + 'I', tiff, 4)[0]
        cnt = struct.unpack_from(en + 'H', tiff, ifd0)[0]
        exif_off = None
        for i in range(cnt):
            p = ifd0 + 2 + i * 12
            tag = struct.unpack_from(en + 'H', tiff, p)[0]
            if tag == 0x8769:
                exif_off = struct.unpack_from(en + 'I', tiff, p + 8)[0]
        if exif_off is None:
            return None
        n2 = struct.unpack_from(en + 'H', tiff, exif_off)[0]
        mn_off = None
        for j in range(n2):
            q = exif_off + 2 + j * 12
            if struct.unpack_from(en + 'H', tiff, q)[0] == 0x927C:
                mn_off = struct.unpack_from(en + 'I', tiff, q + 8)[0]
                break
        if mn_off is None or not tiff[mn_off:].startswith(b'SONY'):
            return None
        mn = tiff[mn_off:]
        n = struct.unpack_from('<H', mn, 12)[0]
        for i in range(n):
            p = 14 + i * 12
            if p + 12 > len(mn):
                break
            tag, typ, c = struct.unpack_from('<HHI', mn, p)
            if tag == SONY_FOCUS_LOCATION and typ == 3 and c == 4:
                vo = struct.unpack_from('<I', mn, p + 8)[0]
                v = struct.unpack_from('<4H', tiff, vo)
                if 0 < v[0] < 30000 and 0 < v[1] < 30000 and 0 < v[2] < v[0] and 0 < v[3] < v[1]:
                    return v[0], v[1], v[2], v[3]
    except Exception:
        pass
    return None


def _expected_pixel(fw, fh, x, y, orient, w, h):
    nx, ny = x / float(fw), y / float(fh)
    if orient == 8:
        return ny * w, (1.0 - nx) * h
    if orient == 6:
        return (1.0 - ny) * w, nx * h
    if orient == 3:
        return (1.0 - nx) * w, (1.0 - ny) * h
    return nx * w, ny * h


def main():
    full = '--full' in sys.argv
    files = []
    for d in sorted(glob.glob('E:/DCIM/*/')):
        files += sorted(glob.glob(d + '*.JPG'))
    if not files:
        print('未找到测试照片（期望 E:/DCIM/*/*.JPG）')
        return
    sample = files if full else files[::max(1, len(files) // 120)][:120]

    ok = matched = checked = 0
    fails = []
    for f in sample:
        try:
            with open(f, 'rb') as fh:
                data = fh.read(250000)
            tiff = _tiff_of(data)
            if tiff is None:
                continue
            ind = _sony_focus_independent(tiff)
            if ind is None:
                continue
            checked += 1
            orient = _orientation(tiff)

            boxes = album.extract_af_points(f, orient)
            if not boxes:
                fails.append((os.path.basename(f), 'no box'))
                continue
            ok += 1

            # 复刻 paintEvent 换算
            r = boxes[0]['rect']
            import cv2
            img = cv2.imdecode(__import__('numpy').fromfile(f, dtype='uint8'),
                               cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]
            cx = (r.x() + r.width() / 2) * w
            cy = (r.y() + r.height() / 2) * h
            ex, ey = _expected_pixel(ind[0], ind[1], ind[2], ind[3], orient, w, h)
            tol = r.width() * w
            if abs(cx - ex) <= tol and abs(cy - ey) <= tol:
                matched += 1
            else:
                fails.append((os.path.basename(f), orient,
                              (round(cx), round(cy)), (round(ex), round(ey))))
        except Exception as e:
            fails.append((os.path.basename(f), f'ERR {type(e).__name__}: {e}'))

    print(f'样本 {len(sample)} 张 / 有效索尼对焦点 {checked} 张')
    print(f'产出对焦点框: {ok}/{checked}')
    print(f'与独立解析一致: {matched}/{checked}')
    if fails:
        print('不一致:')
        for x in fails[:10]:
            print('  ', x)
    else:
        print('\n全部通过 ✓')


if __name__ == '__main__':
    main()