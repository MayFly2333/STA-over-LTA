# this function reads a set of data from my local folder, transforming these .mat files to sec in DASPy

import os
import re
import h5py
import numpy as np
from daspy import Section

def safe_name(name: str) -> str:
    # 去掉扩展名
    name = os.path.splitext(name)[0]
    # 把非字母数字字符替换成下划线
    name = re.sub(r'\W+', '_', name)
    return "sec_" + name

# 输入文件夹路径
folder = input("请输入文件夹路径：").strip()

for fname in os.listdir(folder):
    if fname.endswith(".mat"):
        filepath = os.path.join(folder, fname)

        with h5py.File(filepath, "r") as f:
            if "phase" not in f:
                print(f"⚠️ {fname} 中没有 'phase' 变量，跳过。")
                continue
            phase = np.array(f["phase"])

        metadata = {
            "dx": 10.0,
            "fs": 100.0,
            "gauge_length": None,
            "start_channel": 0,
            "start_distance": 0.0,
            "start_time": 0.0,
            "scale": 1.0,
            "source": fname
        }

        sec_name = safe_name(fname)
        globals()[sec_name] = Section(phase, **metadata)  # 直接放进全局变量

        print(f"✅ 已生成 {sec_name}")

print("\n处理完成！现在可以直接用 sec_xxx.plot() 调用了。")
