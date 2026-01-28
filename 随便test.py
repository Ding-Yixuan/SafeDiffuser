import h5py
import os
import numpy as np

file_path = os.path.expanduser('~/.d4rl/datasets/maze2d-custom-v1-dense.hdf5')
print(f"正在检查文件: {file_path}")

def print_structure(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"📄 [数据] {name}: shape={obj.shape}, dtype={obj.dtype}")
        # 特别检查 timeouts
        if 'timeouts' in name:
            data = obj[:]
            print(f"    ↳ True的数量: {np.sum(data)} / {len(data)}")
    elif isinstance(obj, h5py.Group):
        print(f"📂 [文件夹] {name}")

try:
    with h5py.File(file_path, 'r') as f:
        print("\n=== 文件内部结构 ===")
        f.visititems(print_structure)
        
        if 'timeouts' in f:
            print("\n✅ 确认: 文件里包含 'timeouts' 字段！")
        else:
            print("\n❌ 警告: 文件里没有 'timeouts'！")

except Exception as e:
    print(f"\n❌ 出错: {e}")