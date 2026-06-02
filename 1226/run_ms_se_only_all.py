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

# 动态添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from sta_ms_se_only import sta_net_ms_se_only

# ==================== GPU 环境配置 ====================
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            # 限制显存，防止 OOM
            tf.config.set_logical_device_configuration(
                gpu,
                [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])
        try:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
        except Exception:
            pass
except Exception:
    pass

# ==================== 辅助类 ====================
class TargetAccCallback(keras.callbacks.Callback):
    def __init__(self, target_acc):
        super().__init__()
        self.target_acc = target_acc

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        class_output_loss = logs.get("class_output_loss")
        if class_output_loss is not None and class_output_loss <= self.target_acc:
            self.model.stop_training = True

def run_subject(subject_id: int, args: argparse.Namespace):
    # 路径设置
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    data_dir = os.path.join(project_root, "data", "model_input")
    results_dir = os.path.join(current_dir, "results", "ms_se_only_all")
    os.makedirs(results_dir, exist_ok=True)

    subject_file = f"subject_{subject_id:02d}_MI.npz"
    subject_filepath = os.path.join(data_dir, subject_file)

    if not os.path.exists(subject_filepath):
        subject_filepath = f"D:/test/stanet_diy/data/model_input/{subject_file}"
    
    if not os.path.exists(subject_filepath):
        raise FileNotFoundError(f"Data not found: {subject_filepath}")

    with np.load(subject_filepath) as data:
        eeg = data["eeg"]
        fnirs = data["fnirs"]
        label = data["label"]

    fnirs *= 1e3
    label = label.astype(float)

    fold_results = []

    for session in range(3):
        print(f"--- Session {session} [Subject {subject_id:02d}] ---")
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

        # 使用 Dataset
        def create_dataset(eeg_data, fnirs_data, label_data):
            x = {"eeg_input": eeg_data, "fnirs_input": fnirs_data}
            y = {"class_output": label_data, "eeg_output": label_data}
            return tf.data.Dataset.from_tensor_slices((x, y))

        train_ds = create_dataset(eeg_train, fnirs_train, label_train).shuffle(buffer_size=len(eeg_train)).batch(args.batch_size).prefetch(1)
        val_ds = create_dataset(eeg_val, fnirs_val, label_val).batch(args.batch_size).prefetch(1)
        train2_ds = create_dataset(all_eeg, all_fnirs, all_label).shuffle(buffer_size=len(all_eeg)).batch(args.batch_size).prefetch(1)
        test_ds = create_dataset(eeg_test, fnirs_test, label_test).batch(args.batch_size).prefetch(1)

        tf.keras.backend.clear_session()
        
        model = sta_net_ms_se_only()

        model.compile(
            optimizer="adam",
            loss={
                "class_output": tf.keras.losses.SparseCategoricalCrossentropy(),
                "eeg_output": tf.keras.losses.SparseCategoricalCrossentropy(),
            },
            metrics={"class_output": ["accuracy"], "eeg_output": ["accuracy"]},
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_class_output_loss", patience=args.patience, restore_best_weights=True, verbose=0, mode="min")
        ]

        first_history = model.fit(train_ds, epochs=args.first_stage_epochs, verbose=0, validation_data=val_ds, callbacks=callbacks)

        min_loss = min(first_history.history["val_class_output_loss"])
        min_epoch = first_history.history["val_class_output_loss"].index(min_loss)
        target_acc_val = first_history.history["class_output_loss"][min_epoch]

        callbacks_stage2 = [TargetAccCallback(target_acc_val)]
        model.fit(train2_ds, epochs=args.second_stage_epochs, verbose=0, callbacks=callbacks_stage2)

        test_results = model.evaluate(test_ds, verbose=0)
        
        acc_idx = model.metrics_names.index('class_output_accuracy')
        print(f"  -> Acc: {test_results[acc_idx]:.4f}")
        
        fold_results.append(test_results)
        del train_ds, val_ds, train2_ds, test_ds, model
        gc.collect()

    min_len = min(len(fr) for fr in fold_results)
    fold_results = [fr[:min_len] for fr in fold_results]
    mean_results = np.mean(np.array(fold_results), axis=0).tolist()

    out_path = os.path.join(results_dir, f"subject_{subject_id:02d}_ms_se_only.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "subject": subject_id,
            "model": "MS-SE-Net (No IB)",
            "fold_results": fold_results,
            "mean_results": mean_results,
            "params": vars(args)
        }, f, indent=2)

    return mean_results[3] # class_acc

def main():
    class Args:
        def __init__(self):
            self.batch_size = 16
            self.first_stage_epochs = 300
            self.second_stage_epochs = 200
            self.patience = 100
            self.seed = 42

    args = Args()
    
    summary = []
    print(f"🚀 MS-SE-Net (No IB) 全员测试启动")
    
    noib_summary_path = os.path.abspath(os.path.join(current_dir, "../1224/results/summary_1224.csv"))
    if not os.path.exists(noib_summary_path):
        print(f"❌ 未找到 NoIB 结果文件: {noib_summary_path}")
        return
    noib_df = pd.read_csv(noib_summary_path)
    
    summary_csv_path = os.path.join(current_dir, "results", "summary_ms_se_only_vs_noib.csv")
    
    # 断点续传
    start_subject = 1
    if os.path.exists(summary_csv_path):
        try:
            existing_df = pd.read_csv(summary_csv_path)
            if len(existing_df) > 0:
                finished_subjects = existing_df['subject'].tolist()
                summary = existing_df.to_dict('records')
                print(f"🔄 已完成 Subject: {finished_subjects}")
            else:
                finished_subjects = []
        except:
            finished_subjects = []
    else:
        finished_subjects = []

    for subject_id in range(1, 30):
        if subject_id in finished_subjects:
            continue
            
        print(f"\nProcessing Subject {subject_id:02d}...")
        try:
            acc = run_subject(subject_id, args)
            print(f"✅ Subject {subject_id:02d} Done. Acc: {acc:.4f}")
            
            noib_acc = noib_df[noib_df['subject'] == subject_id]['noib_mean'].iloc[0]
            
            summary.append({
                "subject": subject_id,
                "noib_acc": noib_acc,
                "ms_se_only_acc": acc,
                "diff": acc - noib_acc
            })
            
            summary_df = pd.DataFrame(summary)
            summary_df.to_csv(summary_csv_path, index=False)
            
        except Exception as e:
            print(f"❌ Subject {subject_id:02d} Failed: {e}")
            import traceback
            traceback.print_exc()
            
    if summary:
        summary_df = pd.DataFrame(summary)
        avg_noib = summary_df['noib_acc'].mean()
        avg_ms_se_only = summary_df['ms_se_only_acc'].mean()
        
        print("\n" + "=" * 60)
        print("📊 最终对决: MS-SE-Net (No IB) vs NoIB")
        print("=" * 60)
        print(summary_df.to_string(index=False))
        print("-" * 60)
        print(f"AVERAGE | {avg_noib:.4f}     | {avg_ms_se_only:.4f}     | {avg_ms_se_only - avg_noib:+.4f}")
        print(f"\n🎉 All Done. 结果已保存至: {summary_csv_path}")

if __name__ == "__main__":
    main()