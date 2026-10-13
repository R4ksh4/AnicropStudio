# -*- coding: utf-8 -*-
"""debug_arw.py - 索尼 ARW 深度 X 光透视探针"""
import sys
import struct
from pathlib import Path

def x_ray_arw(file_path: str):
    p = Path(file_path)
    if not p.exists():
        print(f"❌ 文件不存在: {file_path}")
        return

    print(f"🔬 正在对索尼生肉执行 X 光扫描: {p.name}")
    with open(p, 'rb') as f:
        data = f.read(4 * 1024 * 1024)  # 扫前 4MB 足够覆盖全部元数据区

    print(f"• 文件头: {data[:8]}")

    # 1. 物理全盘硬搜 b'SONY' 签名
    sony_pos = []
    pos = 0
    while True:
        idx = data.find(b'SONY', pos)
        if idx == -1 or len(sony_pos) > 10:
            break
        sony_pos.append(idx)
        pos = idx + 4
    print(f"• 前 4MB 中 b'SONY' 出现的绝对偏移量: {[hex(x) for x in sony_pos]}")

    for sp in sony_pos[:5]:
        preview = data[sp : sp + 32]
        print(f"  -- @ 0x{sp:X}: {preview}")

    # 2. 物理扫描 0x2027 (对焦点标签) 结构
    # 0x2027 在条目中表现为: \x27\x20\x03\x00\x04\x00\x00\x00 (Tag=0x2027, Type=3 SHORT, Count=4)
    target_tag = b'\x27\x20\x03\x00\x04\x00\x00\x00'
    tag_hits = []
    pos = 0
    while True:
        idx = data.find(target_tag, pos)
        if idx == -1:
            break
        tag_hits.append(idx)
        pos = idx + 8
    print(f"\n• 命中 0x2027 (FocusLocation) 标签结构: {[hex(x) for x in tag_hits]}")

    for tp in tag_hits:
        val_off = struct.unpack('<I', data[tp + 8 : tp + 12])[0]
        print(f"  -- 标签位于 0x{tp:X}, 指向值偏移: 0x{val_off:X} ({val_off})")
        if val_off + 8 <= len(data):
            v = struct.unpack('<4H', data[val_off : val_off + 8])
            print(f"     -> 读取出 4 个值: {v} (full_w, full_h, x, y)")
        # 顺便测一下如果是相对当前 MakerNote 起点的偏移
        for sp in sony_pos:
            rel_off = sp + val_off
            if rel_off + 8 <= len(data):
                v_rel = struct.unpack('<4H', data[rel_off : rel_off + 8])
                print(f"     -> [若相对 SONY@0x{sp:X}] 读取: {v_rel}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        x_ray_arw(sys.argv[1])