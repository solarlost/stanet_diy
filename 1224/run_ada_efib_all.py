import os
import json
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
import math
import gc
import sys
import shutil

# 动态添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from ada_efib import ada_efib_net

# ==================== GPU 环境配置 ====================
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
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

class BetaScheduler(keras.callbacks.Callback):
    def __init__(self, beta_start, beta_end, warmup_epochs):
        super().__init__()
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.warmup_epochs = max(1, int(warmup_epochs))

    def on_epoch_begin(self, epoch, logs=None):
        if epoch >= self.warmup_epochs:
            beta_now = self.beta_end
        else:
            progress = epoch / float(self.warmup_epochs)
            beta_now = self.beta_end - (self.beta_end - self.beta_start) * 0.5 * (1.0 + math.cos(math.pi * progress))

        for layer in self.model.layers:
            # 递归查找子层
            if hasattr(layer, "layers"):
                for sublayer in layer.layers:
                    if hasattr(sublayer, "beta"):
                        try: sublayer.beta = float(beta_now)
                        except: pass
            if hasattr(layer, "beta"):
                try: layer.beta = float(beta_now)
                except: pass

def run_subject(subject_id: int, args: argparse.Namespace):
    # 路径设置
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    data_dir = os.path.join(project_root, "data", "model_input")
    results_dir = os.path.join(current_dir, "results", "ada_efib")
    os.makedirs(results_dir, exist_ok=True)

    subject_file = f"subject_{subject_id:02d}_MI.npz"
    subject_filepath = os.path.join(data_dir, subject_file)

    if not os.path.exists(subject_filepath):
        subject_filepath = f"D:/test/stanet_diy/data/model_input/{subject_file}"
    
    if not os.path.exists(subject_filepath):
        print(f"❌ Data not found: {subject_filepath}")
        return 0.0

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

        with tf.device('/GPU:0'):
            x_train = {"eeg_input": tf.constant(eeg_train, tf.float32), "fnirs_input": tf.constant(fnirs_train, tf.float32)}
            y_train = {"class_output": tf.constant(label_train, tf.float32), "eeg_output": tf.constant(label_train, tf.float32)}
            x_val = {"eeg_input": tf.constant(eeg_val, tf.float32), "fnirs_input": tf.constant(fnirs_val, tf.float32)}
            y_val = {"class_output": tf.constant(label_val, tf.float32), "eeg_output": tf.constant(label_val, tf.float32)}
            x_train2 = {"eeg_input": tf.constant(all_eeg, tf.float32), "fnirs_input": tf.constant(all_fnirs, tf.float32)}
            y_train2 = {"class_output": tf.constant(all_label, tf.float32), "eeg_output": tf.constant(all_label, tf.float32)}
            x_test = {"eeg_input": tf.constant(eeg_test, tf.float32), "fnirs_input": tf.constant(fnirs_test, tf.float32)}
            y_test = {"class_output": tf.constant(label_test, tf.float32), "eeg_output": tf.constant(label_test, tf.float32)}

        tf.keras.backend.clear_session()
        
        # 构建 Ada-EF-IB 模型
        model = ada_efib_net(latent_dim=args.latent_dim, beta=args.beta, align_weight=args.align_weight)

        model.compile(
            optimizer="adam",
            steps_per_execution=10,
            jit_compile=False,
            loss={
                "class_output": tf.keras.losses.SparseCategoricalCrossentropy(),
                "eeg_output": tf.keras.losses.SparseCategoricalCrossentropy(),
            },
            metrics={"class_output": ["accuracy"], "eeg_output": ["accuracy"]},
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_class_output_loss", patience=args.patience, restore_best_weights=True, verbose=0, mode="min"),
            BetaScheduler(args.beta_start, args.beta, args.beta_warmup_epochs)
        ]

        first_history = model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.first_stage_epochs, verbose=0, validation_data=(x_val, y_val), callbacks=callbacks, shuffle=True)

        min_loss = min(first_history.history["val_class_output_loss"])
        min_epoch = first_history.history["val_class_output_loss"].index(min_loss)
        target_acc_val = first_history.history["class_output_loss"][min_epoch]

        callbacks_stage2 = [TargetAccCallback(target_acc_val), BetaScheduler(args.beta_start, args.beta, args.beta_warmup_epochs)]
        model.fit(x_train2, y_train2, batch_size=args.batch_size, epochs=args.second_stage_epochs, verbose=0, callbacks=callbacks_stage2, shuffle=True)

        test_results = model.evaluate(x_test, y_test, batch_size=args.batch_size, verbose=0)
        
        # 打印门控值 (如果可能)
        # 我们可以创建一个临时模型来获取 gate 的输出，但这比较麻烦
        # 简单起见，我们只看最终结果
        
        acc_idx = model.metrics_names.index('class_output_accuracy')
        print(f"  -> Acc: {test_results[acc_idx]:.4f}")
        
        fold_results.append(test_results)
        del x_train, y_train, x_val, y_val, x_train2, y_train2, x_test, y_test, model
        gc.collect()

    min_len = min(len(fr) for fr in fold_results)
    fold_results = [fr[:min_len] for fr in fold_results]
    mean_results = np.mean(np.array(fold_results), axis=0).tolist()

    out_path = os.path.join(results_dir, f"subject_{subject_id:02d}_ada.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "subject": subject_id,
            "model": "Ada-EF-IB-Net",
            "fold_results": fold_results,
            "mean_results": mean_results,
            "params": vars(args)
        }, f, indent=2)

    return mean_results[3] # class_acc

def main():
    # 参数配置
    class Args:
        def __init__(self):
            self.batch_size = 128
            self.first_stage_epochs = 300
            self.second_stage_epochs = 200
            self.patience = 100
            self.seed = 42
            self.beta = 1e-4
            self.beta_start = 1e-4
            self.beta_warmup_epochs = 30
            self.latent_dim = 128
            self.align_weight = 0.001

    args = Args()
    
    summary = []
    print(f"🚀 Ada-EF-IB 全员测试启动")
    
    for subject_id in range(1, 30):
        print(f"\nProcessing Subject {subject_id:02d}...")
        try:
            acc = run_subject(subject_id, args)
            print(f"✅ Subject {subject_id:02d} Done. Acc: {acc:.4f}")
            summary.append({"subject": subject_id, "acc": acc})
        except Exception as e:
            print(f"❌ Subject {subject_id:02d} Failed: {e}")
            import traceback
            traceback.print_exc()
            
    # 保存汇总
    csv_path = os.path.join(current_dir, "results", "summary_ada_all.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("subject,acc\n")
        for item in summary:
            f.write(f"{item['subject']},{item['acc']:.6f}\n")
            
    avg_acc = sum(x['acc'] for x in summary) / len(summary) if summary else 0
    print(f"\n🎉 All Done. Average Acc: {avg_acc:.4f}")

if __name__ == "__main__":
    main()