import os
import json
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
import math
import gc
import sys

# 动态添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from sta_parallel_ib import sta_net_parallel_ib

# ==================== GPU 环境配置 ====================
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            # 限制显存
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
    results_dir = os.path.join(current_dir, "results_ma") # MA 结果单独放
    os.makedirs(results_dir, exist_ok=True)

    # 关键修改: 读取 MA 数据
    subject_file = f"subject_{subject_id:02d}_MA.npz"
    subject_filepath = os.path.join(data_dir, subject_file)

    print(f"📂 尝试读取 MA 数据: {subject_filepath}")

    if not os.path.exists(subject_filepath):
        # 尝试绝对路径
        subject_filepath = f"D:/test/stanet_diy/data/model_input/{subject_file}"
    
    if not os.path.exists(subject_filepath):
        raise FileNotFoundError(f"❌ Data not found: {subject_filepath}")

    with np.load(subject_filepath) as data:
        eeg = data["eeg"]
        fnirs = data["fnirs"]
        label = data["label"]

    fnirs *= 1e3
    label = label.astype(float)

    fold_results = []

    for session in range(3):
        print(f"--- Session {session} [Subject {subject_id:02d} MA] ---")
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
        
        # 构建 Parallel IB Net
        model = sta_net_parallel_ib(latent_dim=args.latent_dim, beta=args.beta)

        model.compile(
            optimizer="adam",
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

        first_history = model.fit(train_ds, epochs=args.first_stage_epochs, verbose=0, validation_data=val_ds, callbacks=callbacks)

        min_loss = min(first_history.history["val_class_output_loss"])
        min_epoch = first_history.history["val_class_output_loss"].index(min_loss)
        target_acc_val = first_history.history["class_output_loss"][min_epoch]

        callbacks_stage2 = [TargetAccCallback(target_acc_val), BetaScheduler(args.beta_start, args.beta, args.beta_warmup_epochs)]
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

    out_path = os.path.join(results_dir, f"subject_{subject_id:02d}_parallel_ib_ma.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "subject": subject_id,
            "task": "MA",
            "model": "MS-SE-Parallel-IB-Net",
            "fold_results": fold_results,
            "mean_results": mean_results,
            "params": vars(args)
        }, f, indent=2)

    return mean_results[3] # class_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=1) # 默认跑 Subject 01
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--first-stage-epochs", type=int, default=300)
    parser.add_argument("--second-stage-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    
    parser.add_argument("--beta", type=float, default=1e-5, help="IB weight")
    parser.add_argument("--beta_start", type=float, default=1e-6)
    parser.add_argument("--beta_warmup_epochs", type=int, default=30)
    parser.add_argument("--latent_dim", type=int, default=128)

    args = parser.parse_args()
    
    print(f"🚀 MS-SE-Parallel-IB-Net 测试 (Subject {args.subject} - MA Task)")
    print(f"⚙️ 参数: Beta={args.beta}")
    
    try:
        acc = run_subject(args.subject, args)
        print(f"\n✅ 测试完成!")
        print(f"🏆 Mean Accuracy: {acc:.4f}")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()