import os
import json
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import math
import gc
import sys
import pandas as pd

# 动态添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 复用 sta_parallel_ib.py 中的组件
from sta_parallel_ib import SEBlock, MultiScaleConv3D, InformationBottleneck, reduce_sum_layer, expand_dims_layer

# ==================== 单模态模型定义 ====================

def sta_net_eeg_only(latent_dim=128, beta=1e-5):
    eeg_input = keras.Input(shape=(16, 16, 600, 1), name="eeg_input")
    
    # 1. MS-SE Encoder
    eeg_feat = MultiScaleConv3D(16, (2, 2), (2, 2, 6), padding='same')(eeg_input)
    eeg_feat = layers.Dropout(0.5)(eeg_feat)
    eeg_feat = MultiScaleConv3D(32, (2, 2), (2, 2, 2), padding='same')(eeg_feat)
    eeg_feat = layers.GlobalAveragePooling3D()(eeg_feat) # Flatten
    eeg_feat = layers.Dense(256, activation='elu')(eeg_feat)
    
    # 2. Parallel IB
    eeg_latent = InformationBottleneck(latent_dim=latent_dim, beta=beta, name='eeg_ib')(eeg_feat)
    eeg_combined = layers.Concatenate()([eeg_feat, eeg_latent])
    
    # 3. Classifier
    x = layers.Dense(64, activation='elu')(eeg_combined)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(2, activation='softmax', name="class_output")(x)
    
    model = keras.Model(inputs=eeg_input, outputs=output, name="eeg_only_net")
    return model

def sta_net_fnirs_only(latent_dim=128, beta=1e-5):
    fnirs_input = keras.Input(shape=(11, 16, 16, 30, 2), name="fnirs_input")
    
    # 1. Reshape
    fnirs_reshaped = tf.reshape(fnirs_input, [-1, 11, 16, 16, 60])
    
    # 2. MS-SE Encoder
    fnirs_feat = MultiScaleConv3D(16, (2, 2), (2, 2, 6), padding='same')(fnirs_reshaped)
    fnirs_feat = layers.Dropout(0.5)(fnirs_feat)
    fnirs_feat = MultiScaleConv3D(32, (2, 2), (2, 2, 2), padding='same')(fnirs_feat)
    fnirs_feat = layers.GlobalAveragePooling3D()(fnirs_feat)
    fnirs_feat = layers.Dense(256, activation='elu')(fnirs_feat)
    
    # 3. Parallel IB
    fnirs_latent = InformationBottleneck(latent_dim=latent_dim, beta=beta, name='fnirs_ib')(fnirs_feat)
    fnirs_combined = layers.Concatenate()([fnirs_feat, fnirs_latent])
    
    # 4. Classifier
    x = layers.Dense(64, activation='elu')(fnirs_combined)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(2, activation='softmax', name="class_output")(x)
    
    model = keras.Model(inputs=fnirs_input, outputs=output, name="fnirs_only_net")
    return model

# ==================== 训练逻辑 ====================

try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.set_logical_device_configuration(gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])
        try: tf.keras.mixed_precision.set_global_policy('mixed_float16')
        except: pass
except Exception: pass

class TargetAccCallback(keras.callbacks.Callback):
    def __init__(self, target_acc):
        super().__init__()
        self.target_acc = target_acc
    def on_epoch_end(self, epoch, logs=None):
        if logs.get("val_loss") is not None and logs.get("val_loss") <= self.target_acc:
            self.model.stop_training = True

class BetaScheduler(keras.callbacks.Callback):
    def __init__(self, beta_start, beta_end, warmup_epochs):
        super().__init__()
        self.beta_start, self.beta_end, self.warmup_epochs = beta_start, beta_end, warmup_epochs
    def on_epoch_begin(self, epoch, logs=None):
        if epoch >= self.warmup_epochs: beta_now = self.beta_end
        else:
            progress = epoch / float(self.warmup_epochs)
            beta_now = self.beta_end - (self.beta_end - self.beta_start) * 0.5 * (1.0 + math.cos(math.pi * progress))
        for layer in self.model.layers:
            if hasattr(layer, "beta"): layer.beta = float(beta_now)

def run_subject_modality(subject_id: int, args: argparse.Namespace, modality: str):
    # 路径
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    data_dir = os.path.join(project_root, "data", "model_input")
    results_dir = os.path.join(current_dir, "results", f"{modality}_only_all")
    os.makedirs(results_dir, exist_ok=True)
    
    # 权重保存目录
    weights_dir = os.path.join(results_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    subject_file = f"subject_{subject_id:02d}_MI.npz"
    subject_filepath = os.path.join(data_dir, subject_file)
    if not os.path.exists(subject_filepath): return None

    with np.load(subject_filepath) as data:
        eeg = data["eeg"]
        fnirs = data["fnirs"]
        label = data["label"]
    fnirs *= 1e3
    label = label.astype(float)

    fold_results = []

    for session in range(3):
        print(f"  >> Session {session} [{modality.upper()}]")
        session_slice = slice(session * 200, (session + 1) * 200)
        
        if modality == "eeg":
            x_all = np.delete(eeg, session_slice, axis=0)
            x_test = eeg[session_slice]
        else:
            x_all = np.delete(fnirs, session_slice, axis=0)
            x_test = fnirs[session_slice]
            
        y_all = np.delete(label, session_slice, axis=0)
        y_test = label[session_slice]

        np.random.seed(args.seed)
        indices = np.random.choice(x_all.shape[0], size=80, replace=False)
        
        x_train = np.delete(x_all, indices, axis=0)
        y_train = np.delete(y_all, indices, axis=0)
        x_val = x_all[indices]
        y_val = y_all[indices]

        def create_dataset(x, y):
            return tf.data.Dataset.from_tensor_slices((x, y))

        train_ds = create_dataset(x_train, y_train).shuffle(len(x_train)).batch(args.batch_size).prefetch(1)
        val_ds = create_dataset(x_val, y_val).batch(args.batch_size).prefetch(1)
        train2_ds = create_dataset(x_all, y_all).shuffle(len(x_all)).batch(args.batch_size).prefetch(1)
        test_ds = create_dataset(x_test, y_test).batch(args.batch_size).prefetch(1)

        tf.keras.backend.clear_session()
        
        if modality == "eeg":
            model = sta_net_eeg_only(latent_dim=args.latent_dim, beta=args.beta)
        else:
            model = sta_net_fnirs_only(latent_dim=args.latent_dim, beta=args.beta)

        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        # 权重保存路径
        weight_path = os.path.join(weights_dir, f"weights_sub{subject_id:02d}_sess{session}_{modality}.h5")

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True, verbose=0, mode="min"),
            BetaScheduler(args.beta_start, args.beta, args.beta_warmup_epochs),
            # 新增: 保存最佳权重
            tf.keras.callbacks.ModelCheckpoint(filepath=weight_path, monitor="val_loss", save_best_only=True, save_weights_only=True, mode="min", verbose=0)
        ]

        first_history = model.fit(train_ds, epochs=args.first_stage_epochs, verbose=0, validation_data=val_ds, callbacks=callbacks)
        
        min_loss = min(first_history.history["val_loss"])
        
        callbacks_stage2 = [TargetAccCallback(min_loss), BetaScheduler(args.beta_start, args.beta, args.beta_warmup_epochs)]
        model.fit(train2_ds, epochs=args.second_stage_epochs, verbose=0, callbacks=callbacks_stage2)
        
        # 保存 Stage 2 结束后的权重
        model.save_weights(weight_path)

        test_results = model.evaluate(test_ds, verbose=0)
        print(f"     Acc: {test_results[1]:.4f}")
        fold_results.append(test_results)
        
        del train_ds, val_ds, train2_ds, test_ds, model
        gc.collect()

    mean_results = np.mean(np.array(fold_results), axis=0).tolist()
    
    out_path = os.path.join(results_dir, f"subject_{subject_id:02d}_{modality}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"subject": subject_id, "modality": modality, "mean_results": mean_results}, f)

    return mean_results[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--first-stage-epochs", type=int, default=300)
    parser.add_argument("--second-stage-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta", type=float, default=1e-5)
    parser.add_argument("--beta_start", type=float, default=1e-6)
    parser.add_argument("--beta_warmup_epochs", type=int, default=30)
    parser.add_argument("--latent_dim", type=int, default=128)
    args = parser.parse_args()
    
    summary = []
    print(f"🚀 单模态全员测试启动 (EEG & fNIRS) - 带权重保存")
    
    summary_csv_path = os.path.join(current_dir, "results", "summary_unimodal.csv")
    
    # 断点续传
    start_subject = 1
    if os.path.exists(summary_csv_path):
        try:
            existing_df = pd.read_csv(summary_csv_path)
            if len(existing_df) > 0:
                finished_subjects = existing_df['subject'].tolist()
                summary = existing_df.to_dict('records')
                print(f"🔄 已完成: {finished_subjects}")
            else: finished_subjects = []
        except: finished_subjects = []
    else: finished_subjects = []

    for subject_id in range(1, 30):
        if subject_id in finished_subjects: continue
        
        print(f"\nProcessing Subject {subject_id:02d}...")
        try:
            # 1. Run EEG
            acc_eeg = run_subject_modality(subject_id, args, "eeg")
            
            # 2. Run fNIRS
            acc_fnirs = run_subject_modality(subject_id, args, "fnirs")
            
            if acc_eeg is not None and acc_fnirs is not None:
                print(f"✅ Subject {subject_id:02d} Done. EEG: {acc_eeg:.4f}, fNIRS: {acc_fnirs:.4f}")
                summary.append({
                    "subject": subject_id,
                    "eeg_acc": acc_eeg,
                    "fnirs_acc": acc_fnirs
                })
                pd.DataFrame(summary).to_csv(summary_csv_path, index=False)
        except Exception as e:
            print(f"❌ Failed: {e}")
            import traceback
            traceback.print_exc()
            
    if summary:
        df = pd.DataFrame(summary)
        print(f"\n📊 Average Acc: EEG={df['eeg_acc'].mean():.4f}, fNIRS={df['fnirs_acc'].mean():.4f}")

if __name__ == "__main__":
    main()