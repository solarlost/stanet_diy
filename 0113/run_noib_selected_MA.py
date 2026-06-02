import os
import json
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
import math
import gc
import sys
import pandas as pd

# 动态添加路径: 引用 1224 文件夹下的 sta_net (NoIB)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(os.path.join(project_root, "STA-NET"))

from sta import sta_net # 引用原版 STA-Net

# ==================== GPU 环境配置 ====================
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
        if logs.get("class_output_loss") is not None and logs.get("class_output_loss") <= self.target_acc:
            self.model.stop_training = True

def run_subject(subject_id: int, args: argparse.Namespace):
    data_dir = os.path.join(project_root, "data", "model_input")
    subject_file = f"subject_{subject_id:02d}_MA.npz" # MA 数据
    subject_filepath = os.path.join(data_dir, subject_file)

    if not os.path.exists(subject_filepath):
        print(f"⚠️ Data not found: {subject_filepath}")
        return None

    with np.load(subject_filepath) as data:
        eeg = data["eeg"]
        fnirs = data["fnirs"]
        label = data["label"]
    fnirs *= 1e3
    label = label.astype(float)

    fold_results = []

    for session in range(3):
        print(f"--- Session {session} [Subject {subject_id:02d} MA - NoIB] ---")
        session_slice = slice(session * 200, (session + 1) * 200)
        all_eeg = np.delete(eeg, session_slice, axis=0)
        all_fnirs = np.delete(fnirs, session_slice, axis=0)
        all_label = np.delete(label, session_slice, axis=0)
        eeg_test = eeg[session_slice]
        fnirs_test = fnirs[session_slice]
        label_test = label[session_slice]

        np.random.seed(args.seed)
        indices = np.random.choice(all_eeg.shape[0], size=80, replace=False)
        eeg_train = np.delete(all_eeg, indices, axis=0)
        fnirs_train = np.delete(all_fnirs, indices, axis=0)
        label_train = np.delete(all_label, indices, axis=0)
        eeg_val = all_eeg[indices]
        fnirs_val = all_fnirs[indices]
        label_val = all_label[indices]

        def create_dataset(eeg_data, fnirs_data, label_data):
            x = {"eeg_input": eeg_data, "fnirs_input": fnirs_data}
            y = {"class_output": label_data, "eeg_output": label_data}
            return tf.data.Dataset.from_tensor_slices((x, y))

        train_ds = create_dataset(eeg_train, fnirs_train, label_train).shuffle(len(eeg_train)).batch(args.batch_size).prefetch(1)
        val_ds = create_dataset(eeg_val, fnirs_val, label_val).batch(args.batch_size).prefetch(1)
        train2_ds = create_dataset(all_eeg, all_fnirs, all_label).shuffle(len(all_eeg)).batch(args.batch_size).prefetch(1)
        test_ds = create_dataset(eeg_test, fnirs_test, label_test).batch(args.batch_size).prefetch(1)

        tf.keras.backend.clear_session()
        model = sta_net() # NoIB Model

        model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])

        callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_class_output_loss", patience=args.patience, restore_best_weights=True, verbose=0, mode="min")]
        first_history = model.fit(train_ds, epochs=args.first_stage_epochs, verbose=0, validation_data=val_ds, callbacks=callbacks)
        
        min_loss = min(first_history.history["val_class_output_loss"])
        callbacks_stage2 = [TargetAccCallback(min_loss)]
        model.fit(train2_ds, epochs=args.second_stage_epochs, verbose=0, callbacks=callbacks_stage2)

        test_results = model.evaluate(test_ds, verbose=0)
        print(f"  -> Acc: {test_results[1]:.4f}")
        fold_results.append(test_results)
        
        del train_ds, val_ds, train2_ds, test_ds, model
        gc.collect()

    mean_acc = np.mean([res[1] for res in fold_results])
    return mean_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--first-stage-epochs", type=int, default=300)
    parser.add_argument("--second-stage-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # 挑选的被试
    selected_subjects = [1, 8, 11, 17, 25]
    
    print(f"🚀 STA-Net (NoIB) MA 抽样测试: {selected_subjects}")
    
    results = []
    for sub in selected_subjects:
        print(f"\nProcessing Subject {sub:02d}...")
        try:
            acc = run_subject(sub, args)
            if acc is not None:
                print(f"✅ Subject {sub:02d} Done. Acc: {acc:.4f}")
                results.append({"subject": sub, "noib_ma_acc": acc})
        except Exception as e:
            print(f"❌ Failed: {e}")
            import traceback
            traceback.print_exc()
            
    if results:
        df = pd.DataFrame(results)
        print("\n" + "=" * 40)
        print("📊 STA-Net (NoIB) MA Results")
        print("=" * 40)
        print(df.to_string(index=False))
        print("-" * 40)
        print(f"Average: {df['noib_ma_acc'].mean():.4f}")

if __name__ == "__main__":
    main()