import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import gc

# --- 动态添加路径以导入模型 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(os.path.join(project_root, "1226"))

from sta_ms_se_only import sta_net_ms_se_only
from sta_ms_se import sta_net_ms_se as sta_net_ms_se_ib

# --- GPU 配置 ---
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])
        print(f"✅ GPU configured with 4GB memory limit.")
except Exception as e:
    print(f"❌ GPU config error: {e}")

def get_features(model_builder, subject_id, use_ib, seed=42):
    """
    训练一个模型并提取其倒数第二层的特征。
    """
    print(f"\n--- Processing model: {'with IB' if use_ib else 'without IB'} ---")
    
    # --- 1. 加载数据 (仅使用第一折) ---
    data_dir = os.path.join(project_root, "data", "model_input")
    subject_file = f"subject_{subject_id:02d}_MI.npz"
    subject_filepath = os.path.join(data_dir, subject_file)
    with np.load(subject_filepath) as data:
        eeg = data["eeg"]
        fnirs = data["fnirs"]
        label = data["label"]
    fnirs *= 1e3
    label = label.astype(float)

    session_slice = slice(0, 200) # Test on session 0
    eeg_train = np.delete(eeg, session_slice, axis=0)
    fnirs_train = np.delete(fnirs, session_slice, axis=0)
    label_train = np.delete(label, session_slice, axis=0)
    eeg_test = eeg[session_slice]
    fnirs_test = fnirs[session_slice]
    label_test = label[session_slice]

    # --- 2. 训练模型 ---
    tf.keras.backend.clear_session()
    
    # 构建完整模型
    if use_ib:
        model = model_builder(latent_dim=128, beta=1e-5)
    else:
        model = model_builder()

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    
    print("  -> Training model...")
    # 简化训练，不分阶段，直接 fit
    model.fit(
        {"eeg_input": eeg_train, "fnirs_input": fnirs_train},
        {"class_output": label_train, "eeg_output": label_train},
        batch_size=16,
        epochs=100, # 减少 epoch 数量以加快速度
        verbose=0,
        shuffle=True
    )
    
    # --- 3. 构建特征提取器 ---
    if use_ib:
        # 提取三个 IB 层的输出
        layer_names = ['eegfusion_ib', 'fnirs_ib', 'eeg_ib']
    else:
        # 提取三个 Activation 层的输出 (我们在 sta_ms_se_only.py 中命名的)
        layer_names = ['fusion_feature_output', 'fnirs_feature_output', 'eeg_feature_output']

    print(f"  -> Extracting features from layers: {layer_names}")
    try:
        output_layers = [model.get_layer(name).output for name in layer_names]
        feature_extractor = keras.Model(inputs=model.inputs, outputs=output_layers)
    except ValueError as e:
        print(f"❌ Layer not found: {e}")
        print("Available layers:", [layer.name for layer in model.layers])
        raise e

    print("  -> Extracting features...")
    test_data = {"eeg_input": eeg_test, "fnirs_input": fnirs_test}
    extracted_features = feature_extractor.predict(test_data, batch_size=16)
    
    # 将三个分支的特征拼接起来
    combined_features = np.concatenate(extracted_features, axis=1)
    
    del model, feature_extractor
    gc.collect()
    
    return combined_features, label_test

def main():
    SUBJECT_ID = 9 # S9 在 MS-SE-IB-Net 上表现很好
    
    # --- 获取特征 ---
    features_no_ib, labels = get_features(sta_net_ms_se_only, SUBJECT_ID, use_ib=False)
    features_with_ib, _ = get_features(sta_net_ms_se_ib, SUBJECT_ID, use_ib=True)
    
    # --- t-SNE 降维 ---
    print("\n--- Running t-SNE (this may take a while)... ---")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    
    print("  -> Transforming features without IB...")
    tsne_no_ib = tsne.fit_transform(features_no_ib)
    
    print("  -> Transforming features with IB...")
    tsne_with_ib = tsne.fit_transform(features_with_ib)
    
    # --- 绘图 ---
    print("\n--- Plotting results... ---")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # 颜色
    colors = ['#1f77b4', '#ff7f0e'] # Blue for class 0, Orange for class 1
    
    # 图 1: Without IB
    for i in range(2): # 两个类别
        idx = labels == i
        ax1.scatter(tsne_no_ib[idx, 0], tsne_no_ib[idx, 1], c=colors[i], label=f'Class {i}', alpha=0.7)
    ax1.set_title('t-SNE of Features (MS-SE-Net without IB)', fontsize=14)
    ax1.set_xlabel('t-SNE Dimension 1')
    ax1.set_ylabel('t-SNE Dimension 2')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    # 图 2: With IB
    for i in range(2):
        idx = labels == i
        ax2.scatter(tsne_with_ib[idx, 0], tsne_with_ib[idx, 1], c=colors[i], label=f'Class {i}', alpha=0.7)
    ax2.set_title('t-SNE of Features (MS-SE-IB-Net)', fontsize=14)
    ax2.set_xlabel('t-SNE Dimension 1')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # 保存图像
    output_path = os.path.join(current_dir, "tsne_comparison_s09.png")
    plt.savefig(output_path)
    print(f"\n✅ t-SNE comparison plot saved to: {output_path}")

if __name__ == "__main__":
    main()