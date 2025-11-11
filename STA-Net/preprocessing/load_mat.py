import scipy.io as io
import numpy as np
import os

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 项目根目录：从 STA-Net/preprocessing/ 向上两级到项目根目录
project_root = os.path.dirname(os.path.dirname(script_dir))

# 数据路径配置
eeg_base_path = os.path.join(project_root, 'data', 'EEG')
fnirs_base_path = os.path.join(project_root, 'data', 'FNIRS')
save_dir = os.path.join(project_root, 'data', 'mat2array')

# 检查输入目录是否存在
if not os.path.exists(eeg_base_path):
    raise FileNotFoundError(f"EEG数据目录不存在: {eeg_base_path}\n请确保数据目录结构为: data/EEG/subject XX/")
if not os.path.exists(fnirs_base_path):
    raise FileNotFoundError(f"FNIRS数据目录不存在: {fnirs_base_path}\n请确保数据目录结构为: data/FNIRS/subject XX/")

# 创建输出目录
os.makedirs(save_dir, exist_ok=True)

# 获取被试列表
subject_list = []
eeg_folders = [f for f in os.listdir(eeg_base_path) if os.path.isdir(os.path.join(eeg_base_path, f)) and f.startswith('subject')]
subject_list = sorted(eeg_folders, key=lambda x: int(x.split()[-1]))

print(f"找到 {len(subject_list)} 个被试")
print(f"被试列表: {subject_list}")

for subject_folder in subject_list:
    subject_name = subject_folder.replace(' ', '_')  # subject 01 -> subject_01
    
    # EEG文件路径 - 使用"with occular artifact"文件夹中的文件
    eeg_cnt_path = os.path.join(eeg_base_path, subject_folder, 'with occular artifact', 'cnt.mat')
    eeg_mrk_path = os.path.join(eeg_base_path, subject_folder, 'with occular artifact', 'mrk.mat')
    
    # FNIRS文件路径
    fnirs_cnt_path = os.path.join(fnirs_base_path, subject_folder, 'cnt.mat')
    fnirs_mrk_path = os.path.join(fnirs_base_path, subject_folder, 'mrk.mat')
    
    # 检查文件是否存在
    if not os.path.exists(eeg_cnt_path):
        print(f"警告: EEG文件不存在: {eeg_cnt_path}")
        continue
    if not os.path.exists(eeg_mrk_path):
        print(f"警告: EEG marker文件不存在: {eeg_mrk_path}")
        continue
    if not os.path.exists(fnirs_cnt_path):
        print(f"警告: FNIRS文件不存在: {fnirs_cnt_path}")
        continue
    if not os.path.exists(fnirs_mrk_path):
        print(f"警告: FNIRS marker文件不存在: {fnirs_mrk_path}")
        continue
    
    try:
        # 加载MATLAB文件（使用struct_as_record=False和squeeze_me=True来正确访问结构体）
        eeg_data = io.loadmat(eeg_cnt_path, struct_as_record=False, squeeze_me=True)
        eeg_mrk_data = io.loadmat(eeg_mrk_path, struct_as_record=False, squeeze_me=True)
        fnirs_data = io.loadmat(fnirs_cnt_path, struct_as_record=False, squeeze_me=True)
        fnirs_mrk_data = io.loadmat(fnirs_mrk_path, struct_as_record=False, squeeze_me=True)
        
        # 访问EEG数据
        # cnt[0].x 是EEG数据，shape (timepoints, channels)，需要转置为 (channels, timepoints)
        eeg = eeg_data['cnt'][0].x.T
        
        # 访问EEG marker数据
        # mrk[0].time 是时间标记
        eeg_time = eeg_mrk_data['mrk'][0].time
        # mrk[0].y 是标签，shape (2, n_events)，需要转置为 (n_events, 2) 或取argmax
        label = eeg_mrk_data['mrk'][0].y.T  # shape (n_events, 2)
        # 将one-hot编码转换为类别标签（0或1）
        label = np.argmax(label, axis=1)  # shape (n_events,)
        
        # 访问FNIRS数据
        # cnt[0].x 是FNIRS数据，shape (timepoints, channels)
        # 前36个通道（0-35）是lowWL（760nm，HBR - deoxy）
        # 后36个通道（36-71）是highWL（850nm，HBO - oxy）
        fnirs_x = fnirs_data['cnt'][0].x  # shape (timepoints, 72)
        hbr = fnirs_x[:, :36].T  # 前36个通道，转置为 (channels, timepoints)
        hbo = fnirs_x[:, 36:].T  # 后36个通道，转置为 (channels, timepoints)
        
        # 访问FNIRS marker数据
        fnirs_time = fnirs_mrk_data['mrk'][0].time
        
        print(f"\n处理 {subject_folder}:")
        print(f"  EEG shape: {eeg.shape}")
        print(f"  EEG time shape: {eeg_time.shape}")
        print(f"  HBO shape: {hbo.shape}")
        print(f"  HBR shape: {hbr.shape}")
        print(f"  FNIRS time shape: {fnirs_time.shape}")
        print(f"  Label shape: {label.shape}")
        
        save_dict = {
            'eeg':eeg,
            'eeg_time':eeg_time,
            'hbo':hbo,
            'hbr':hbr,
            'fnirs_time':fnirs_time,
            'label':label
        }
        
        save_name = subject_name
        save_path = os.path.join(save_dir, save_name)
        
        np.savez(save_path, **save_dict)
        print(f'==============保存 {save_name} 成功=============\n')
        
    except Exception as e:
        print(f'错误: 处理 {subject_folder} 时出错: {str(e)}')
        import traceback
        traceback.print_exc()
        continue

print(f'\n所有处理完成！输出目录: {save_dir}')







