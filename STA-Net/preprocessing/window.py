import numpy as np
import os

'''
win_step = 1
win_length = 3

eeg_segments_number = 10
fnirs_segments_number = 22

eeg_srate = 200
fnirs_srate = 10

subject_path = r'E:\IF\review\new_dataset\d3'
subject_list = os.listdir(subject_path)

for subject in subject_list:
    with np.load(os.path.join(subject_path, subject)) as data:
        eeg = data['eeg']
        hbo = data['hbo']
        hbr = data['hbr']
        label = data['label']

    eeg_window = np.ones((60, eeg_segments_number, 16, 16, win_length*eeg_srate))
    hbo_window = np.ones((60, fnirs_segments_number, 16, 16, win_length*fnirs_srate))
    hbr_window = np.ones((60, fnirs_segments_number, 16, 16, win_length*fnirs_srate))

    for e in range(60):
        # first 10 windows has same time interval
        for w in range(eeg_segments_number):
            eeg_start_indice = (3+w)*eeg_srate
            eeg_end_indice = eeg_start_indice + win_length*eeg_srate

            eeg_segment = eeg[e, :, :, eeg_start_indice:eeg_end_indice]

            eeg_window[e, w, :, :, :] = eeg_segment

        for fw in range(fnirs_segments_number):
            fnirs_start_indice = (3+fw)*fnirs_srate
            fnirs_end_indice = fnirs_start_indice + win_length*fnirs_srate

            hbo_segment = hbo[e, :, :, fnirs_start_indice:fnirs_end_indice]
            hbr_segment = hbr[e, :, :, fnirs_start_indice:fnirs_end_indice]

            hbo_window[e, fw, :, :, :] = hbo_segment
            hbr_window[e, fw, :, :, :] = hbr_segment

    print(eeg_window.shape)
    print(hbo_window.shape)
    print(hbr_window.shape)
    print(label.shape)
    
    save_dict = {
    'eeg':eeg_window,
    'hbo':hbo_window,
    'hbr':hbr_window,
    'label':label
    }

    save_dir = r'E:\IF\review\new_dataset\window'
    save_name = subject

    np.savez(os.path.join(save_dir,save_name),**save_dict)
    print('\n==============save {} success=============\n'.format(save_name)) 
'''



fnirs_lag_length = 11 # with t-self

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 项目根目录：从 STA-Net/preprocessing/ 向上两级到项目根目录
project_root = os.path.dirname(os.path.dirname(script_dir))

# 路径配置
# 注意：window.py需要先运行一个步骤从d3创建window数据，然后再运行当前步骤
# 如果window目录不存在，则从d3目录读取并创建window数据
d3_path = os.path.join(project_root, 'data', 'd3')
window_path = os.path.join(project_root, 'data', 'window')
save_dir = os.path.join(project_root, 'data', 'model_input')

# 创建输出目录
os.makedirs(window_path, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)

# 检查window目录是否有数据，如果没有则从d3创建
if not os.path.exists(window_path) or len([f for f in os.listdir(window_path) if f.endswith('.npz')]) == 0:
    print("window目录为空，从d3目录创建window数据...")
    # 这里需要先运行window创建步骤（注释掉的代码部分）
    # 为了简化，我们直接从d3读取并处理
    subject_path = d3_path
else:
    subject_path = window_path

subject_list = [f for f in os.listdir(subject_path) if f.endswith('.npz')]
subject_list = sorted(subject_list)

print(f"找到 {len(subject_list)} 个被试文件")
print(f"开始窗口处理...")

for subject in subject_list:
    with np.load(os.path.join(subject_path, subject)) as data:
        eeg = data['eeg']
        hbo = data['hbo']
        hbr = data['hbr']
        label = data['label']
    
    n_epochs = eeg.shape[0]
    n_windows = 10  # 每个epoch的窗口数
    
    # 计算EEG窗口：每个窗口600个时间点（3秒 * 200Hz）
    # 从时间点3*200=600开始，取600个时间点
    eeg_win_length = 600
    eeg_start = 600  # 3秒 * 200Hz
    
    # 计算fNIRS窗口：每个窗口30个时间点（3秒 * 10Hz）
    fnirs_win_length = 30
    fnirs_start = 30  # 3秒 * 10Hz
    
    # 检查数据长度是否足够
    if eeg.shape[-1] < eeg_start + eeg_win_length:
        print(f"警告: {subject} 的EEG数据长度不足，跳过")
        continue
    if hbo.shape[-1] < fnirs_start + fnirs_win_length:
        print(f"警告: {subject} 的fNIRS数据长度不足，跳过")
        continue
    
    # EEG处理：每个epoch生成10个窗口
    eeg_windows = []
    for e in range(n_epochs):
        for w in range(n_windows):
            eeg_start_idx = eeg_start + w * 200  # 每个窗口间隔1秒（200个采样点）
            eeg_end_idx = eeg_start_idx + eeg_win_length
            if eeg_end_idx > eeg.shape[-1]:
                # 如果超出范围，使用最后一个窗口
                eeg_end_idx = eeg.shape[-1]
                eeg_start_idx = eeg_end_idx - eeg_win_length
            eeg_window = eeg[e, :, :, eeg_start_idx:eeg_end_idx]
            eeg_windows.append(eeg_window)
    
    eeg_input = np.array(eeg_windows)  # shape: (n_epochs * n_windows, 16, 16, 600)
    eeg_input = np.expand_dims(eeg_input, axis=-1)  # shape: (n_epochs * n_windows, 16, 16, 600, 1)
    
    # fNIRS处理：每个epoch生成10个窗口，每个窗口使用11个时间步的滞后
    fnirs_windows = []
    for e in range(n_epochs):
        for w in range(n_windows):
            # fNIRS窗口从时间点30开始，每个窗口间隔10个采样点（1秒）
            fnirs_start_idx = fnirs_start + w * 10
            fnirs_end_idx = fnirs_start_idx + fnirs_win_length
            if fnirs_end_idx > hbo.shape[-1]:
                fnirs_end_idx = hbo.shape[-1]
                fnirs_start_idx = fnirs_end_idx - fnirs_win_length
            
            # 创建滞后窗口：使用11个时间步
            fnirs_lag_windows = []
            for lag in range(fnirs_lag_length):
                lag_start = fnirs_start_idx - lag
                lag_end = lag_start + fnirs_win_length
                if lag_start < 0:
                    # 如果滞后窗口超出范围，使用零填充
                    hbo_lag = np.zeros((16, 16, fnirs_win_length))
                    hbr_lag = np.zeros((16, 16, fnirs_win_length))
                else:
                    hbo_lag = hbo[e, :, :, lag_start:lag_end]
                    hbr_lag = hbr[e, :, :, lag_start:lag_end]
                
                # 合并HBO和HBR
                hbo_lag = np.expand_dims(hbo_lag, axis=-1)
                hbr_lag = np.expand_dims(hbr_lag, axis=-1)
                fnirs_lag = np.concatenate((hbo_lag, hbr_lag), axis=-1)
                fnirs_lag_windows.append(fnirs_lag)
            
            # 堆叠11个滞后窗口
            fnirs_window = np.stack(fnirs_lag_windows, axis=0)  # shape: (11, 16, 16, 30, 2)
            fnirs_windows.append(fnirs_window)
    
    fnirs_input = np.array(fnirs_windows)  # shape: (n_epochs * n_windows, 11, 16, 16, 30, 2)
    
    # label处理：每个epoch的标签重复10次
    label_input = np.repeat(label, repeats=n_windows, axis=0)

    print(eeg_input.shape)
    print(fnirs_input.shape)
    print(label_input.shape)

    save_dict = {
    'eeg':eeg_input,
    'fnirs':fnirs_input,
    'label':label_input
    }

    save_name = os.path.splitext(subject)[0]
    save_path = os.path.join(save_dir, save_name)

    np.savez(save_path, **save_dict)
    print('\n==============保存 {} 成功=============\n'.format(save_name))

print(f'\n所有窗口处理完成！输出目录: {save_dir}') 


    


    
            




    