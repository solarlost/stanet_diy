import numpy as np
from scipy.interpolate import griddata
import os
from tqdm import tqdm

# all points
x = np.arange(16)
y = np.arange(16)
xx, yy = np.meshgrid(x, y)
all_points = np.column_stack((xx.ravel(), yy.ravel()))

# eeg interpolate
# [x, y]
known_eeg_point_coordinates = np.array([[0., 6.], #Fp1
                                        [2., 5.], #AFF5h
                                        [2., 8.], #AFz
                                        [3., 7.], #F1
                                        [5., 2.], #FC5
                                        [5., 6.], #FC1
                                        [7., 1.], #T7
                                        [7., 4.], #C3
                                        [7., 8.], #Cz
                                        [9., 2.], #CP5
                                        [9., 6.], #CP1
                                        [11., 2.], #P7
                                        [11., 5.], #P3
                                        [11., 8.], #Pz
                                        [13., 8.], #POz
                                        [14., 6.], #O1
                                        [0., 10.], #Fp2
                                        [2., 11.], #AFF6h
                                        [3., 9.], #F2
                                        [5., 10.], #FC2
                                        [5., 14.], #FC6
                                        [7., 12.], #C4
                                        [7., 15.], #T8
                                        [9., 10.], #CP2
                                        [9., 14.], #CP6
                                        [11., 11.], #P4
                                        [11., 14.], #P8
                                        [14., 10.] #O2
                                        ])

unknown_eeg_point_coordinates = np.array([coord for coord in all_points if coord.tolist() not in known_eeg_point_coordinates.tolist()])
unknown_eeg_point_coordinates = unknown_eeg_point_coordinates.astype(float)

# fnirs interpolate
known_fnirs_point_coordinates = np.array([[2., 4.], #AF7
                                          [3., 4.], #AFF5
                                          [1., 5.], #AFp7
                                          [2., 5.], #AF5h
                                          [1., 7.], #AFp3
                                          [3., 6.], #AFF3h
                                          [2., 7.], #AF1
                                          [3., 8.], #AFFz
                                          [1., 8.], #AFpz
                                          [2., 9.], #AF2
                                          [1., 9.], #AFp4
                                          [6., 4.], #FCC3
                                          [7., 5.], #C3h
                                          [7., 3.], #C5h
                                          [8., 4.], #CCP3
                                          [10., 5.], #CPP3
                                          [11., 6.], #P3h
                                          [11., 4.], #P5h
                                          [12., 5.], #PPO3
                                          [3., 10.], #AFF4h
                                          [2., 11.], #AF6h
                                          [3., 12.], #AFF6
                                          [1., 11.], #AFp8
                                          [2., 12.], #AF8
                                          [6., 12.], #FCC4
                                          [7., 13.], #C6h
                                          [7., 11.], #C4h
                                          [8., 12.], #CCP4
                                          [10., 11.], #CPP4
                                          [11., 12.], #P6h
                                          [11., 10.], #P4h
                                          [12., 11.], #PPO4
                                          [12., 8.], #PPOz
                                          [13., 7.], #PO1
                                          [13., 9.], #PO2
                                          [14., 8.] #POOz
                                        ])

unknown_fnirs_point_coordinates = np.array([coord for coord in all_points if coord.tolist() not in known_fnirs_point_coordinates.tolist()])
unknown_fnirs_point_coordinates = unknown_fnirs_point_coordinates.astype(float)

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 项目根目录：从 STA-Net/preprocessing/ 向上两级到项目根目录
project_root = os.path.dirname(os.path.dirname(script_dir))

# 路径配置
subject_path = os.path.join(project_root, 'data', 'epoch')
save_dir = os.path.join(project_root, 'data', 'd3')

# 创建输出目录
os.makedirs(save_dir, exist_ok=True)

subject_list = [f for f in os.listdir(subject_path) if f.endswith('.npz')]
subject_list = sorted(subject_list)

print(f"找到 {len(subject_list)} 个被试文件")
print(f"开始3D转换处理...")

for subject in tqdm(subject_list, desc="处理被试", unit="文件"):
    with np.load(os.path.join(subject_path, subject)) as data:
        eeg = data['eeg']
        hbo = data['hbo']
        hbr = data['hbr']
        label = data['label']

    eeg_3dtensor = np.ones((eeg.shape[0], 16, 16, eeg.shape[-1]))
    fnirs_hbo_3dtensor = np.ones((hbo.shape[0], 16, 16, hbo.shape[-1]))
    fnirs_hbr_3dtensor = np.ones((hbr.shape[0], 16, 16, hbr.shape[-1]))

    n_epoch = eeg.shape[0]
    assert eeg.shape[0] == hbo.shape[0] == hbr.shape[0], f"数据形状不匹配: eeg={eeg.shape[0]}, hbo={hbo.shape[0]}, hbr={hbr.shape[0]}"
    
    subject_name = os.path.splitext(subject)[0]
    for e in tqdm(range(n_epoch), desc=f"  {subject_name}", leave=False, unit="epoch"):
        # 3D eeg
        for t in range(eeg.shape[-1]):
            known_eeg_point_values = eeg[e, :, t]

            # create 16*16 array
            eeg_2dimage = np.ones((16, 16))

            # cubic spline interpolate
            eeg_interpolated_values = griddata(points=known_eeg_point_coordinates,
                                            values=known_eeg_point_values,
                                            xi=unknown_eeg_point_coordinates,
                                            method='cubic')
            
            # y=row, x=col
            # first known points
            assert known_eeg_point_values.shape[0] == known_eeg_point_coordinates.shape[0] == 28
            for k in range(28):
                eeg_2dimage[int(known_eeg_point_coordinates[k, 0]), int(known_eeg_point_coordinates[k, 1])] = known_eeg_point_values[k]

            # second unknown points
            assert eeg_interpolated_values.shape[0] == unknown_eeg_point_coordinates.shape[0] == 228
            for u in range(228):
                eeg_2dimage[int(unknown_eeg_point_coordinates[u, 0]), int(unknown_eeg_point_coordinates[u, 1])] = eeg_interpolated_values[u]

            # nearest interpolate
            aftcub_known_eeg_point_values = eeg_2dimage[~np.isnan(eeg_2dimage)]
            aftcub_known_eeg_point_coordinates = np.argwhere(~np.isnan(eeg_2dimage))
            nan_eeg_point_coordinates = np.argwhere(np.isnan(eeg_2dimage))

            nan_eeg_interpolated_values = griddata(points=aftcub_known_eeg_point_coordinates,
                                                    values=aftcub_known_eeg_point_values,
                                                    xi=nan_eeg_point_coordinates,
                                                    method='nearest')
            
            for ne in range(nan_eeg_point_coordinates.shape[0]):
                eeg_2dimage[nan_eeg_point_coordinates[ne, 0], nan_eeg_point_coordinates[ne, 1]] = nan_eeg_interpolated_values[ne]

            eeg_3dtensor[e, :, :, t] = eeg_2dimage

        # 3D fnirs
        assert hbo.shape[-1] == hbr.shape[-1]
        for ft in range(hbo.shape[-1]):
            known_hbo_point_values = hbo[e, :, ft]
            known_hbr_point_values = hbr[e, :, ft]

            # create 16*16 array
            hbo_2dimage = np.ones((16, 16))
            hbr_2dimage = np.ones((16, 16))

             # cubic spline interpolate
            hbo_interpolated_values = griddata(points=known_fnirs_point_coordinates,
                                            values=known_hbo_point_values,
                                            xi=unknown_fnirs_point_coordinates,
                                            method='cubic')
            
            hbr_interpolated_values = griddata(points=known_fnirs_point_coordinates,
                                            values=known_hbr_point_values,
                                            xi=unknown_fnirs_point_coordinates,
                                            method='cubic')
            
            # first known points
            assert known_hbo_point_values.shape[0] == known_hbr_point_values.shape[0] == known_fnirs_point_coordinates.shape[0] == 36
            for fk in range(36):
                hbo_2dimage[int(known_fnirs_point_coordinates[fk, 0]), int(known_fnirs_point_coordinates[fk, 1])] = known_hbo_point_values[fk]
                hbr_2dimage[int(known_fnirs_point_coordinates[fk, 0]), int(known_fnirs_point_coordinates[fk, 1])] = known_hbr_point_values[fk]

            # second unknown points
            assert hbo_interpolated_values.shape[0] == hbr_interpolated_values.shape[0] == unknown_fnirs_point_coordinates.shape[0] == 220
            for fu in range(220):
                hbo_2dimage[int(unknown_fnirs_point_coordinates[fu, 0]), int(unknown_fnirs_point_coordinates[fu, 1])] = hbo_interpolated_values[fu]
                hbr_2dimage[int(unknown_fnirs_point_coordinates[fu, 0]), int(unknown_fnirs_point_coordinates[fu, 1])] = hbr_interpolated_values[fu]

            # nearest interpolate
            aftcub_known_hbo_point_values = hbo_2dimage[~np.isnan(hbo_2dimage)]
            aftcub_known_hbo_point_coordinates = np.argwhere(~np.isnan(hbo_2dimage))
            nan_hbo_point_coordinates = np.argwhere(np.isnan(hbo_2dimage))

            aftcub_known_hbr_point_values = hbr_2dimage[~np.isnan(hbr_2dimage)]
            aftcub_known_hbr_point_coordinates = np.argwhere(~np.isnan(hbr_2dimage))
            nan_hbr_point_coordinates = np.argwhere(np.isnan(hbr_2dimage))

            nan_hbo_interpolated_values = griddata(points=aftcub_known_hbo_point_coordinates,
                                                values=aftcub_known_hbo_point_values,
                                                xi=nan_hbo_point_coordinates,
                                                method='nearest')
            
            nan_hbr_interpolated_values = griddata(points=aftcub_known_hbr_point_coordinates,
                                                values=aftcub_known_hbr_point_values,
                                                xi=nan_hbr_point_coordinates,
                                                method='nearest')
            
            assert nan_hbo_point_coordinates.shape[0] == nan_hbr_point_coordinates.shape[0]
            for nf in range(nan_hbo_point_coordinates.shape[0]):
                hbo_2dimage[nan_hbo_point_coordinates[nf, 0], nan_hbo_point_coordinates[nf, 1]] = nan_hbo_interpolated_values[nf]
                hbr_2dimage[nan_hbr_point_coordinates[nf, 0], nan_hbr_point_coordinates[nf, 1]] = nan_hbr_interpolated_values[nf]

            fnirs_hbo_3dtensor[e, :, :, ft] = hbo_2dimage
            fnirs_hbr_3dtensor[e, :, :, ft] = hbr_2dimage

    save_dict = {
            'eeg':eeg_3dtensor,
            'hbo':fnirs_hbo_3dtensor,
            'hbr':fnirs_hbr_3dtensor,
            'label':label
        }
    
    save_name = os.path.splitext(subject)[0]
    save_path = os.path.join(save_dir, save_name)

    np.savez(save_path, **save_dict)

print(f'\n所有3D转换处理完成！共处理 {len(subject_list)} 个文件，输出目录: {save_dir}')