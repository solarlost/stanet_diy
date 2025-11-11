import numpy as np
import os
import mne

task_period = 10

eeg_sample_rate = 200
eeg_pre_onset = 5
eeg_post_onset = task_period

fnirs_sample_rate = 10
fnirs_pre_onset = 5
fnirs_post_onset = task_period + 12

fnirs_chn_names = ['AF7','AFF5','AFp7','AF5h','AFp3','AFF3h','AF1','AFFz','AFpz','AF2','AFp4','FCC3','C3h','C5h','CCP3','CPP3','P3h','P5h','PPO3','AFF4h','AF6h','AFF6','AFp8','AF8','FCC4','C6h','C4h','CCP4','CPP4','P6h','P4h','PPO4','PPOz','PO1','PO2','POOz']
fnirs_info = mne.create_info(ch_names=fnirs_chn_names, sfreq=10, ch_types='eeg')
fnirs_info.set_montage('standard_1005')

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 项目根目录：从 STA-Net/preprocessing/ 向上两级到项目根目录
project_root = os.path.dirname(os.path.dirname(script_dir))

# 路径配置
subject_path = os.path.join(project_root, 'data', 'preprocessed')
save_dir = os.path.join(project_root, 'data', 'epoch')

# 创建输出目录
os.makedirs(save_dir, exist_ok=True)

subject_list = [f for f in os.listdir(subject_path) if f.endswith('.npz')]
subject_list = sorted(subject_list)

print(f"找到 {len(subject_list)} 个文件")
print(f"开始epoch处理...")

for subject in subject_list:
    subject_name = os.path.splitext(subject)[0]
    
    # 检查是否是MI或MA文件
    if not (subject_name.endswith('_MI') or subject_name.endswith('_MA')):
        print(f"跳过非MI/MA文件: {subject_name}")
        continue
    
    print(f"\n处理 {subject_name}...")
    
    try:
        # 加载数据（包含多个run的列表）
        with np.load(os.path.join(subject_path, subject), allow_pickle=True) as data:
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
        
        # 存储所有run的epoch数据
        all_eeg_epochs = []
        all_hbo_epochs = []
        all_hbr_epochs = []
        all_labels = []
        
        # 对每个run独立进行epoch提取
        for run_idx in range(len(eeg_runs)):
            print(f"  处理run {run_idx + 1}/{len(eeg_runs)}...")
            
            eeg = eeg_runs[run_idx]
            eeg_time = eeg_time_runs[run_idx]
            hbo = hbo_runs[run_idx]
            hbr = hbr_runs[run_idx]
            fnirs_time = fnirs_time_runs[run_idx]
            label = label_runs[run_idx]
            
            # 获取该run的事件数量
            n_events = len(eeg_time)
            print(f"    Run {run_idx + 1}: {n_events} 个事件")
            
            # 收集有效的事件
            valid_eeg_epochs = []
            valid_hbo_epochs = []
            valid_hbr_epochs = []
            valid_labels = []
            
            for t in range(n_events):
                # eeg
                eeg_start_indice = int((eeg_time[t]/1000.-eeg_pre_onset)*eeg_sample_rate)
                eeg_end_indice = int(eeg_start_indice + (eeg_pre_onset+eeg_post_onset)*eeg_sample_rate)
                
                # 检查边界，确保不超出run的范围
                if eeg_start_indice < 0:
                    print(f"    警告: 事件 {t} 的EEG起始索引 {eeg_start_indice} < 0，跳过该事件")
                    continue
                if eeg_end_indice > eeg.shape[1]:
                    print(f"    警告: 事件 {t} 的EEG结束索引 {eeg_end_indice} > run长度 {eeg.shape[1]}，跳过该事件")
                    continue
                
                # fnirs
                fnirs_start_indice = int((fnirs_time[t]/1000.-fnirs_pre_onset)*fnirs_sample_rate)
                fnirs_end_indice = int(fnirs_start_indice + (fnirs_pre_onset+fnirs_post_onset)*fnirs_sample_rate)
                
                # 检查边界，确保不超出run的范围
                if fnirs_start_indice < 0:
                    print(f"    警告: 事件 {t} 的FNIRS起始索引 {fnirs_start_indice} < 0，跳过该事件")
                    continue
                if fnirs_end_indice > hbo.shape[1]:
                    print(f"    警告: 事件 {t} 的FNIRS结束索引 {fnirs_end_indice} > run长度 {hbo.shape[1]}，跳过该事件")
                    continue
                
                # 提取epoch数据
                eeg_one_epoch = eeg[:, eeg_start_indice:eeg_end_indice]
                hbo_one_epoch = hbo[:, fnirs_start_indice:fnirs_end_indice]
                hbr_one_epoch = hbr[:, fnirs_start_indice:fnirs_end_indice]
                
                valid_eeg_epochs.append(eeg_one_epoch)
                valid_hbo_epochs.append(hbo_one_epoch)
                valid_hbr_epochs.append(hbr_one_epoch)
                valid_labels.append(label[t])
            
            # 转换为numpy数组
            if len(valid_eeg_epochs) == 0:
                print(f"    警告: Run {run_idx + 1} 没有有效事件，跳过该run")
                continue
            
            eeg_epoch = np.array(valid_eeg_epochs)  # shape (n_valid_events, 28, timepoints)
            hbo_epoch = np.array(valid_hbo_epochs)  # shape (n_valid_events, 36, timepoints)
            hbr_epoch = np.array(valid_hbr_epochs)  # shape (n_valid_events, 36, timepoints)
            label_run_valid = np.array(valid_labels)  # shape (n_valid_events,)
            
            # fnirs baseline correction
            hbo_raw_bc = mne.EpochsArray(data=hbo_epoch, info=fnirs_info, baseline=(None, 3.))
            hbr_raw_bc = mne.EpochsArray(data=hbr_epoch, info=fnirs_info, baseline=(None, 3.))  
            hbo_epoch_bc = hbo_raw_bc.get_data()
            hbr_epoch_bc = hbr_raw_bc.get_data()
            
            # 收集该run的epoch数据
            all_eeg_epochs.append(eeg_epoch)
            all_hbo_epochs.append(hbo_epoch_bc)
            all_hbr_epochs.append(hbr_epoch_bc)
            all_labels.append(label_run_valid)
        
        # 合并所有run的epoch数据
        eeg_epoch_combined = np.concatenate(all_eeg_epochs, axis=0)
        hbo_epoch_combined = np.concatenate(all_hbo_epochs, axis=0)
        hbr_epoch_combined = np.concatenate(all_hbr_epochs, axis=0)
        label_combined = np.concatenate(all_labels, axis=0)
        
        print(f"  合并后: {eeg_epoch_combined.shape[0]} 个事件")
        print(f"  EEG epoch shape: {eeg_epoch_combined.shape}")
        print(f"  HBO epoch shape: {hbo_epoch_combined.shape}")
        print(f"  HBR epoch shape: {hbr_epoch_combined.shape}")
        print(f"  Label shape: {label_combined.shape}")
        
        save_dict = {
            'eeg': eeg_epoch_combined,
            'hbo': hbo_epoch_combined,
            'hbr': hbr_epoch_combined,
            'label': label_combined
        }
        
        save_path = os.path.join(save_dir, subject_name)
        np.savez(save_path, **save_dict)
        print('\n==============保存 {} 成功=============\n'.format(subject_name))
        
    except Exception as e:
        print(f'错误: 处理 {subject_name} 时出错: {str(e)}')
        import traceback
        traceback.print_exc()
        continue

print(f'\n所有epoch处理完成！输出目录: {save_dir}')


