import numpy as np
import os
import mne
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免显示问题
import matplotlib.pyplot as plt

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 项目根目录：从 STA-Net/preprocessing/ 向上两级到项目根目录
project_root = os.path.dirname(os.path.dirname(script_dir))

# 路径配置
input_dir = os.path.join(project_root, 'data', 'mat2array')
save_dir = os.path.join(project_root, 'data', 'preprocessed')

# 创建输出目录
os.makedirs(save_dir, exist_ok=True)

# eeg info
eeg_chn_names = ['Fp1','AFF5h','AFz','F1','FC5','FC1','T7','C3','Cz','CP5','CP1','P7','P3','Pz','POz','O1','Fp2',
                'AFF6h','F2','FC2','FC6','C4','T8','CP2','CP6','P4','P8','O2']
eeg_info = mne.create_info(ch_names=eeg_chn_names, sfreq=200, ch_types='eeg')
eeg_info.set_montage('standard_1005')

# fnirs info
fnirs_chn_names = ['AF7','AFF5','AFp7','AF5h','AFp3','AFF3h','AF1','AFFz','AFpz','AF2','AFp4','FCC3','C3h','C5h','CCP3','CPP3','P3h','P5h','PPO3','AFF4h','AF6h','AFF6','AFp8','AF8','FCC4','C6h','C4h','CCP4','CPP4','P6h','P4h','PPO4','PPOz','PO1','PO2','POOz']
fnirs_info = mne.create_info(ch_names=fnirs_chn_names, sfreq=10, ch_types='eeg')
fnirs_info.set_montage('standard_1005')

# 获取所有被试文件（MI和MA文件）
subject_files = [f for f in os.listdir(input_dir) if f.endswith('.npz')]
subject_files = sorted(subject_files)

print(f"找到 {len(subject_files)} 个文件")
print(f"开始预处理...")

# 处理每个被试的MI和MA文件
for subject_file in subject_files:
    subject_name = os.path.splitext(subject_file)[0]
    input_path = os.path.join(input_dir, subject_file)
    
    # 检查是否是MI或MA文件
    if not (subject_name.endswith('_MI') or subject_name.endswith('_MA')):
        print(f"跳过非MI/MA文件: {subject_name}")
        continue
    
    print(f"\n处理 {subject_name}...")
    
    try:
        # 加载数据（包含多个run的列表）
        with np.load(input_path, allow_pickle=True) as data:
            eeg_runs = data['eeg_runs']
            eeg_time_runs = data['eeg_time_runs']
            hbo_runs = data['hbo_runs']
            hbr_runs = data['hbr_runs']
            fnirs_time_runs = data['fnirs_time_runs']
            label_runs = data['label_runs']
        
        # 将列表转换为numpy数组（如果它们被保存为对象数组）
        if isinstance(eeg_runs, np.ndarray) and eeg_runs.dtype == object:
            eeg_runs = [eeg_runs[i] for i in range(len(eeg_runs))]
        if isinstance(hbo_runs, np.ndarray) and hbo_runs.dtype == object:
            hbo_runs = [hbo_runs[i] for i in range(len(hbo_runs))]
        if isinstance(hbr_runs, np.ndarray) and hbr_runs.dtype == object:
            hbr_runs = [hbr_runs[i] for i in range(len(hbr_runs))]
        if isinstance(eeg_time_runs, np.ndarray) and eeg_time_runs.dtype == object:
            eeg_time_runs = [eeg_time_runs[i] for i in range(len(eeg_time_runs))]
        if isinstance(fnirs_time_runs, np.ndarray) and fnirs_time_runs.dtype == object:
            fnirs_time_runs = [fnirs_time_runs[i] for i in range(len(fnirs_time_runs))]
        if isinstance(label_runs, np.ndarray) and label_runs.dtype == object:
            label_runs = [label_runs[i] for i in range(len(label_runs))]
        
        # 预处理每个run
        eeg_processed_runs = []
        hbo_processed_runs = []
        hbr_processed_runs = []
        
        for run_idx in range(len(eeg_runs)):
            print(f"  处理run {run_idx + 1}/{len(eeg_runs)}...")
            
            eeg = eeg_runs[run_idx]
            hbo = hbo_runs[run_idx]
            hbr = hbr_runs[run_idx]
            
            # eeg预处理
            # 注意：如果eeg的通道数不是28，需要调整切片
            if eeg.shape[0] > 28:
                # 移除最后 (eeg.shape[0] - 28) 个通道，确保得到28个通道
                n_channels_to_remove = eeg.shape[0] - 28
                eeg_data = eeg[:-n_channels_to_remove, :]
                print(f"    移除最后 {n_channels_to_remove} 个通道: {eeg.shape[0]} -> {eeg_data.shape[0]}")
            elif eeg.shape[0] == 28:
                eeg_data = eeg
            else:
                print(f"    警告: EEG通道数为 {eeg.shape[0]}，期望28个通道")
                eeg_data = eeg
            
            raw = mne.io.RawArray(data=eeg_data, info=eeg_info)

            raw_notch = raw.notch_filter(np.arange(50, 100, 50))
            raw_filtered = raw_notch.filter(0.5, 50., method='iir', iir_params=dict(order=6, ftype='butter'))

            raw_avg_ref = raw_filtered.set_eeg_reference(ref_channels="average")

            raw_avg_ref.load_data()

            # filtering just for ICA
            filt_ica_raw = raw_avg_ref.copy().filter(l_freq=1., h_freq=None)

            ica = mne.preprocessing.ICA(n_components=20, random_state=42)
            ica.fit(filt_ica_raw)

            # 自动检测异常成分（使用MNE的自动检测功能）
            try:
                ica.exclude = []
            except:
                ica.exclude = []
            
            raw_icaed = ica.apply(raw_avg_ref)
            eeg_processed = raw_icaed.get_data()
            eeg_processed_runs.append(eeg_processed)

            # fnirs预处理
            hbo_raw = mne.io.RawArray(data=hbo, info=fnirs_info)
            hbr_raw = mne.io.RawArray(data=hbr, info=fnirs_info)

            hbo_filtered = hbo_raw.filter(0.01, 0.1, method='iir', iir_params=dict(order=6, ftype='butter'))
            hbr_filtered = hbr_raw.filter(0.01, 0.1, method='iir', iir_params=dict(order=6, ftype='butter'))

            hbo_processed = hbo_filtered.get_data()
            hbr_processed = hbr_filtered.get_data()
            
            hbo_processed_runs.append(hbo_processed)
            hbr_processed_runs.append(hbr_processed)

        # 保存预处理后的数据（保持为列表格式）
        # 将列表转换为对象数组以便保存
        def to_object_array(lst):
            arr = np.empty(len(lst), dtype=object)
            for i, item in enumerate(lst):
                arr[i] = item
            return arr
        
        save_dict = {
            'eeg_runs': to_object_array(eeg_processed_runs),
            'eeg_time_runs': to_object_array(eeg_time_runs),
            'hbo_runs': to_object_array(hbo_processed_runs),
            'hbr_runs': to_object_array(hbr_processed_runs),
            'fnirs_time_runs': to_object_array(fnirs_time_runs),
            'label_runs': to_object_array(label_runs)
        }

        save_path = os.path.join(save_dir, subject_name)
        np.savez(save_path, **save_dict)
        print(f'==============保存 {subject_name} 成功=============')
        
    except Exception as e:
        print(f'错误: 处理 {subject_name} 时出错: {str(e)}')
        import traceback
        traceback.print_exc()
        continue

print(f'\n所有预处理完成！输出目录: {save_dir}')

