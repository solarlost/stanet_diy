import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

# 动态添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(os.path.join(project_root, "0113")) # 引用 0113 文件夹下的模型

from sta_parallel_ib import sta_net_parallel_ib

# GPU 配置
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])
except Exception: pass

def plot_cm(subject_id, model_builder, class_names=['Left Hand', 'Right Hand']):
    print(f"\n--- Processing Subject {subject_id:02d} ---")
    
    # 1. 加载数据
    data_dir = os.path.join(project_root, "data", "model_input")
    subject_file = f"subject_{subject_id:02d}_MI.npz"
    subject_filepath = os.path.join(data_dir, subject_file)
    
    with np.load(subject_filepath) as data:
        eeg = data["eeg"]
        fnirs = data["fnirs"]
        label = data["label"]
    fnirs *= 1e3
    label = label.astype(float)

    # 使用第一折作为测试集 (Session 0)
    session_slice = slice(0, 200)
    eeg_train = np.delete(eeg, session_slice, axis=0)
    fnirs_train = np.delete(fnirs, session_slice, axis=0)
    label_train = np.delete(label, session_slice, axis=0)
    eeg_test = eeg[session_slice]
    fnirs_test = fnirs[session_slice]
    label_test = label[session_slice]

    # 2. 训练模型 (快速训练)
    tf.keras.backend.clear_session()
    model = model_builder(latent_dim=128, beta=1e-5)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    print("  -> Training model (this may take a few minutes)...")
    # 简单训练 100 epoch，不做复杂验证，只为出图
    model.fit(
        {"eeg_input": eeg_train, "fnirs_input": fnirs_train},
        {"class_output": label_train, "eeg_output": label_train},
        batch_size=16,
        epochs=100,
        verbose=0,
        shuffle=True
    )
    
    # 3. 预测
    print("  -> Predicting...")
    preds = model.predict({"eeg_input": eeg_test, "fnirs_input": fnirs_test}, verbose=0)
    # preds[0] 是 class_output
    y_pred = np.argmax(preds[0], axis=1)
    y_true = label_test.astype(int)
    
    # 4. 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    acc = np.trace(cm) / np.sum(cm)
    print(f"  -> Accuracy: {acc:.4f}")
    
    # 5. 绘图
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 16})
    plt.title(f'Confusion Matrix - Subject {subject_id:02d}\n(Acc: {acc:.2%})', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    output_path = os.path.join(current_dir, f"cm_subject_{subject_id:02d}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved to: {output_path}")
    plt.close()

def main():
    # 绘制 S25 (高分) 和 S02 (IB 提升显著)
    plot_cm(25, sta_net_parallel_ib)
    plot_cm(2, sta_net_parallel_ib)

if __name__ == "__main__":
    main()