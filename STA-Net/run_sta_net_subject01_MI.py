from sta import sta_net

import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import gc  # 用于手动垃圾回收

# ==================== 1. GPU 基础配置 ====================
# 这一步是为了防止 TensorFlow 一启动就占满显存导致报错
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ 检测到 {len(gpus)} 个 GPU，已开启显存按需分配模式。")
        print(f"   设备列表: {gpus}")
    except RuntimeError as e:
        print(f"❌ GPU 设置错误: {e}")
else:
    print("⚠️ 未检测到 GPU，将使用 CPU 进行训练（速度会慢）。")
# ========================================================

# 路径配置
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
subject_path = os.path.join(project_root, 'data', 'model_input')
subject_file = 'subject_01_MI.npz'
subject_filepath = os.path.join(subject_path, subject_file)

if not os.path.exists(subject_filepath):
    raise FileNotFoundError(f"目标数据文件不存在: {subject_filepath}")

print("开始训练第一个被试（MI）数据集")
print(f"加载文件: {subject_file}")


# 自定义回调函数：达到目标精度即停止
class TargetAccCallback(keras.callbacks.Callback):
    def __init__(self, target_acc):
        super().__init__()
        self.target_acc = target_acc

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        class_output_loss = logs.get('class_output_loss')
        if class_output_loss is not None and class_output_loss <= self.target_acc:
            print(f"\nReached target loss value {self.target_acc:.4f}; cancelling training!\n")
            self.model.stop_training = True


# 加载并预处理数据
with np.load(subject_filepath) as data:
    eeg = data['eeg']
    fnirs = data['fnirs']
    label = data['label']

fnirs *= 1e3
label = label.astype(float)

n_samples = eeg.shape[0]
if n_samples < 600:
    raise ValueError(f"{subject_file} 仅包含 {n_samples} 个样本，无法执行 3 折交叉验证。")

# ==================== 2. 全局超参数设置 ====================
BATCH_SIZE = 128  # RTX 4070 显存大，开大 Batch Size 提升利用率
# ==========================================================

for session in range(3):
    print(f"\n{'=' * 20} 被试: subject_01_MI, 会话: {session} {'=' * 20}")

    # --- A. 数据切分 (CPU 上进行) ---
    session_slice = slice(session * 200, (session + 1) * 200)

    # 1. 剩下的所有数据 (用于 Second Train)
    all_eeg = np.delete(eeg, session_slice, axis=0)
    all_fnirs = np.delete(fnirs, session_slice, axis=0)
    all_label = np.delete(label, session_slice, axis=0)

    # 2. 测试集
    eeg_test = eeg[session_slice]
    fnirs_test = fnirs[session_slice]
    label_test = label[session_slice]

    # 3. 从剩下的数据中分出 验证集 (80个) 和 训练集 (320个)
    np.random.seed(42)
    indices = np.random.choice(all_eeg.shape[0], size=80, replace=False)

    eeg_train = np.delete(all_eeg, indices, axis=0)
    fnirs_train = np.delete(all_fnirs, indices, axis=0)
    label_train = np.delete(all_label, indices, axis=0)

    eeg_val = all_eeg[indices]
    fnirs_val = all_fnirs[indices]
    label_val = all_label[indices]

    if session == 0:
        print(f"数据形状确认:")
        print(f"  Train: EEG {eeg_train.shape}, fNIRS {fnirs_train.shape}")
        print(f"  Val:   EEG {eeg_val.shape}")
        print(f"  Test:  EEG {eeg_test.shape}")

    # --- B. 核弹级优化：直接将数据上传到 GPU 显存 ---
    # 这步做完后，CPU 就不再负责传数据了，完全是 GPU 内部自嗨，速度极快。
    print("正在将数据一次性全部搬运至 GPU 显存...")

    try:
        with tf.device('/GPU:0'):
            # 训练集 1 (First Train)
            x_train_gpu = {
                "eeg_input": tf.constant(eeg_train, dtype=tf.float32),
                "fnirs_input": tf.constant(fnirs_train, dtype=tf.float32)
            }
            y_train_gpu = {
                "class_output": tf.constant(label_train, dtype=tf.float32),
                "eeg_output": tf.constant(label_train, dtype=tf.float32)
            }

            # 验证集
            x_val_gpu = {
                "eeg_input": tf.constant(eeg_val, dtype=tf.float32),
                "fnirs_input": tf.constant(fnirs_val, dtype=tf.float32)
            }
            y_val_gpu = {
                "class_output": tf.constant(label_val, dtype=tf.float32),
                "eeg_output": tf.constant(label_val, dtype=tf.float32)
            }

            # 训练集 2 (Second Train - 包含 train+val)
            x_train2_gpu = {
                "eeg_input": tf.constant(all_eeg, dtype=tf.float32),
                "fnirs_input": tf.constant(all_fnirs, dtype=tf.float32)
            }
            y_train2_gpu = {
                "class_output": tf.constant(all_label, dtype=tf.float32),
                "eeg_output": tf.constant(all_label, dtype=tf.float32)
            }

            # 测试集
            x_test_gpu = {
                "eeg_input": tf.constant(eeg_test, dtype=tf.float32),
                "fnirs_input": tf.constant(fnirs_test, dtype=tf.float32)
            }
            y_test_gpu = {
                "class_output": tf.constant(label_test, dtype=tf.float32),
                "eeg_output": tf.constant(label_test, dtype=tf.float32)
            }
        print("✅ 数据搬运完成！")
    except RuntimeError as e:
        print(f"❌ 显存不足或 GPU 错误: {e}")
        print("请尝试减小数据量或回退到 Dataset 模式。")
        exit()

    # --- C. 模型构建与编译 ---
    tf.keras.backend.clear_session()
    model = sta_net()

    model.compile(
        optimizer='adam',
        loss={
            'class_output': tf.keras.losses.SparseCategoricalCrossentropy(),
            'eeg_output': tf.keras.losses.SparseCategoricalCrossentropy()
        },
        metrics={
            'class_output': ['accuracy'],
            'eeg_output': ['accuracy']
        },
        # ================== 关键优化参数 ==================
        # steps_per_execution=10: 让 GPU 连续跑 10 个 Batch 再回头找 Python，
        # 对于小数据集，这能减少 90% 的 CPU 通讯开销。
        steps_per_execution=10,

        # jit_compile=False: 关闭 XLA 编译，避免 Conda 环境下 libdevice 缺失的报错
        jit_compile=False
        # ==================================================
    )

    stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_class_output_loss',
        patience=50,
        restore_best_weights=True,
        verbose=1,
        mode='min'
    )

    # --- D. 第一阶段训练 ---
    print('>>> Begin First Train (Train set only)')
    first_history = model.fit(
        x_train_gpu, y_train_gpu,  # 直接传入 GPU Tensor
        batch_size=BATCH_SIZE,
        epochs=300,
        verbose=2,
        validation_data=(x_val_gpu, y_val_gpu),  # 验证集也是 GPU Tensor
        callbacks=[stopping],
        shuffle=True  # 允许打乱
    )

    # 获取最佳 Loss 用于第二阶段停止条件
    min_val_loss = min(first_history.history['val_class_output_loss'])
    min_epoch = first_history.history['val_class_output_loss'].index(min_val_loss)
    target_acc = first_history.history['class_output_loss'][min_epoch]
    print(f"Target Loss determined: {target_acc:.4f}")

    # --- E. 第二阶段训练 ---
    print('>>> Begin Second Train (Train + Val sets)')
    model.fit(
        x_train2_gpu, y_train2_gpu,
        batch_size=BATCH_SIZE,
        epochs=200,
        verbose=2,
        callbacks=[TargetAccCallback(target_acc)],
        shuffle=True
    )

    # --- F. 测试 ---
    print('>>> Begin Test')
    test_results = model.evaluate(x_test_gpu, y_test_gpu, batch_size=BATCH_SIZE)
    print(f"测试结果 [Loss, Class_Loss, EEG_Loss, Class_Acc, EEG_Acc...]:")
    print(test_results)

    # 内存清理：显式删除 GPU 上的 Tensor，防止显存随 Session 累积爆炸
    del x_train_gpu, y_train_gpu, x_val_gpu, y_val_gpu
    del x_train2_gpu, y_train2_gpu, x_test_gpu, y_test_gpu
    gc.collect()

print('\nsubject_01_MI 所有会话训练完成！')