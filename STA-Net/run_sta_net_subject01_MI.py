from sta import sta_net

import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 项目根目录：从 STA-Net/ 向上一级到项目根目录
project_root = os.path.dirname(script_dir)

# 路径配置
subject_path = os.path.join(project_root, 'data', 'model_input')
subject_file = 'subject_01_MI.npz'
subject_filepath = os.path.join(subject_path, subject_file)

# 检查路径是否存在
if not os.path.exists(subject_filepath):
    raise FileNotFoundError(
        f"目标数据文件不存在: {subject_filepath}\n请确认已经完成预处理并生成该文件。"
    )

print("开始训练第一个被试（MI）数据集")
print(f"加载文件: {subject_file}")


class TargetAccCallback(keras.callbacks.Callback):
    def __init__(self, target_acc):
        super().__init__()
        self.target_acc = target_acc

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        class_output_loss = logs.get('class_output_loss')
        if class_output_loss is not None and class_output_loss <= self.target_acc:
            print(
                f"\nReached target loss value {self.target_acc:.4f}; cancelling training!\n"
            )
            self.model.stop_training = True


with np.load(subject_filepath) as data:
    eeg = data['eeg']
    fnirs = data['fnirs']
    label = data['label']

fnirs *= 1e3
label = label.astype(float)

# 检查数据大小，只有600个样本的文件才能进行3折交叉验证
n_samples = eeg.shape[0]
if n_samples < 600:
    raise ValueError(
        f"{subject_file} 仅包含 {n_samples} 个样本，无法执行 3 折交叉验证。"
    )

for session in range(3):
    session_slice = slice(session * 200, (session + 1) * 200)
    all_eeg = np.delete(eeg, session_slice, axis=0)
    all_fnirs = np.delete(fnirs, session_slice, axis=0)
    all_label = np.delete(label, session_slice, axis=0)

    second_train_dataset = tf.data.Dataset.from_tensor_slices(
        (
            {"eeg_input": all_eeg, "fnirs_input": all_fnirs},
            {"class_output": all_label, 'eeg_output': all_label}
        )
    )
    second_train_dataset = second_train_dataset.shuffle(buffer_size=128).batch(32)

    eeg_test = eeg[session_slice]
    fnirs_test = fnirs[session_slice]
    label_test = label[session_slice]

    test_dataset = tf.data.Dataset.from_tensor_slices(
        (
            {"eeg_input": eeg_test, "fnirs_input": fnirs_test},
            {"class_output": label_test, 'eeg_output': label_test}
        )
    )
    test_dataset = test_dataset.batch(32)

    np.random.seed(42)
    indices = np.random.choice(all_eeg.shape[0], size=80, replace=False)

    eeg_train = np.delete(all_eeg, indices, axis=0)
    fnirs_train = np.delete(all_fnirs, indices, axis=0)
    label_train = np.delete(all_label, indices, axis=0)
    first_train_dataset = tf.data.Dataset.from_tensor_slices(
        (
            {"eeg_input": eeg_train, "fnirs_input": fnirs_train},
            {"class_output": label_train, 'eeg_output': label_train}
        )
    )
    first_train_dataset = first_train_dataset.shuffle(buffer_size=128).batch(32)

    eeg_val = all_eeg[indices]
    fnirs_val = all_fnirs[indices]
    label_val = all_label[indices]
    val_dataset = tf.data.Dataset.from_tensor_slices(
        (
            {"eeg_input": eeg_val, "fnirs_input": fnirs_val},
            {"class_output": label_val, 'eeg_output': label_val}
        )
    )
    val_dataset = val_dataset.batch(32)

    print('eeg_train shape:', eeg_train.shape)
    print('fnirs_train shape:', fnirs_train.shape)
    print('label_train shape:', label_train.shape)

    print('eeg_val shape:', eeg_val.shape)
    print('fnirs_val shape:', fnirs_val.shape)
    print('label_val shape:', label_val.shape)

    print(f"被试: subject_01_MI, 会话: {session}")

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
        }
    )

    stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_class_output_loss',
        patience=50,
        restore_best_weights=True,
        verbose=1,
        mode='min'
    )

    print('begin first train')
    first_history = model.fit(
        first_train_dataset,
        epochs=300,
        verbose=2,
        validation_data=val_dataset,
        callbacks=[stopping]
    )

    min_val_class_output_loss = min(first_history.history['val_class_output_loss'])
    min_val_class_output_loss_epoch = first_history.history['val_class_output_loss'].index(
        min_val_class_output_loss
    )
    target_acc = first_history.history['class_output_loss'][min_val_class_output_loss_epoch]

    print('begin second train')
    model.fit(
        second_train_dataset,
        epochs=200,
        verbose=2,
        callbacks=[TargetAccCallback(target_acc)]
    )

    print('begin test')
    test_results = model.evaluate(test_dataset)
    print(f"测试结果: {test_results}")

print('\nsubject_01_MI 训练完成！')

