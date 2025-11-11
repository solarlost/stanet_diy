import scipy.io as io
import numpy as np

# 检查MATLAB文件结构，查找多个run的信息
eeg_mrk_path = 'data/EEG/subject 01/with occular artifact/mrk.mat'

print("检查 mrk.mat 文件结构:")
eeg_mrk_data = io.loadmat(eeg_mrk_path, struct_as_record=False, squeeze_me=True)
mrk = eeg_mrk_data['mrk'][0]

print(f"  mrk.time shape: {mrk.time.shape}")
print(f"  mrk.y shape: {mrk.y.shape}")

# 检查mrk.event
if hasattr(mrk, 'event'):
    print(f"\n  mrk.event type: {type(mrk.event)}")
    if hasattr(mrk.event, 'desc'):
        print(f"  mrk.event.desc: {mrk.event.desc}")
        if isinstance(mrk.event.desc, np.ndarray):
            print(f"  mrk.event.desc shape: {mrk.event.desc.shape}")
            print(f"  mrk.event.desc: {mrk.event.desc}")

# 检查mrk.orig
if hasattr(mrk, 'orig'):
    print(f"\n  mrk.orig type: {type(mrk.orig)}")
    if hasattr(mrk.orig, 'event'):
        print(f"  mrk.orig.event type: {type(mrk.orig.event)}")
        if hasattr(mrk.orig.event, 'desc'):
            print(f"  mrk.orig.event.desc: {mrk.orig.event.desc}")
            if isinstance(mrk.orig.event.desc, np.ndarray):
                print(f"  mrk.orig.event.desc shape: {mrk.orig.event.desc.shape}")
                print(f"  mrk.orig.event.desc: {mrk.orig.event.desc}")

# 检查是否有多个run文件
import os
eeg_base_path = 'data/EEG/subject 01'
print(f"\n检查 {eeg_base_path} 目录结构:")
for root, dirs, files in os.walk(eeg_base_path):
    for file in files:
        if 'run' in file.lower():
            print(f"  找到run相关文件: {os.path.join(root, file)}")

# 检查FNIRS目录
fnirs_base_path = 'data/FNIRS/subject 01'
print(f"\n检查 {fnirs_base_path} 目录结构:")
if os.path.exists(fnirs_base_path):
    for root, dirs, files in os.walk(fnirs_base_path):
        for file in files:
            if 'run' in file.lower():
                print(f"  找到run相关文件: {os.path.join(root, file)}")



