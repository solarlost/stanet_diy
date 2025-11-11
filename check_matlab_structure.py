import scipy.io as io
import numpy as np

# 检查MATLAB文件结构
eeg_mrk_path = 'data/EEG/subject 01/with occular artifact/mrk.mat'
eeg_cnt_path = 'data/EEG/subject 01/with occular artifact/cnt.mat'

print("检查 mrk.mat 文件:")
eeg_mrk_data = io.loadmat(eeg_mrk_path, struct_as_record=False, squeeze_me=True)
mrk = eeg_mrk_data['mrk'][0]

print(f"  mrk.time shape: {mrk.time.shape}")
print(f"  mrk.y shape: {mrk.y.shape}")
print(f"  mrk.time: {mrk.time}")
print(f"  mrk.y: {mrk.y}")

if hasattr(mrk, 'event'):
    print(f"  mrk.event: {mrk.event}")
    if isinstance(mrk.event, np.ndarray):
        print(f"  mrk.event shape: {mrk.event.shape}")
        print(f"  mrk.event: {mrk.event}")

if hasattr(mrk, 'orig'):
    print(f"  mrk.orig type: {type(mrk.orig)}")
    if hasattr(mrk.orig, '__dict__'):
        print(f"  mrk.orig attributes: {list(mrk.orig.__dict__.keys())}")

print("\n检查 cnt.mat 文件:")
eeg_cnt_data = io.loadmat(eeg_cnt_path, struct_as_record=False, squeeze_me=True)
cnt = eeg_cnt_data['cnt'][0]

print(f"  cnt.x shape: {cnt.x.shape}")
print(f"  cnt attributes: {[x for x in dir(cnt) if not x.startswith('_')]}")

# 检查是否有多个run的信息
print("\n检查是否有多个run:")
print(f"  事件数量: {len(mrk.time)}")
print(f"  数据长度: {cnt.x.shape[0]}")

# 检查FNIRS文件
print("\n检查 FNIRS 文件:")
fnirs_mrk_path = 'data/FNIRS/subject 01/mrk.mat'
fnirs_cnt_path = 'data/FNIRS/subject 01/cnt.mat'

fnirs_mrk_data = io.loadmat(fnirs_mrk_path, struct_as_record=False, squeeze_me=True)
fnirs_mrk = fnirs_mrk_data['mrk'][0]

print(f"  fnirs_mrk.time shape: {fnirs_mrk.time.shape}")
print(f"  fnirs_mrk.time: {fnirs_mrk.time}")

fnirs_cnt_data = io.loadmat(fnirs_cnt_path, struct_as_record=False, squeeze_me=True)
fnirs_cnt = fnirs_cnt_data['cnt'][0]

print(f"  fnirs_cnt.x shape: {fnirs_cnt.x.shape}")



