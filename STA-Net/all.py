import os
import json
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
import math
import gc

from sta import sta_net, sta_net_ib

# ==================== GPU 环境配置 ====================
# 在脚本最开始执行，确保环境正确
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # 开启混合精度加速 (RTX 4070 强项)
        try:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
        except Exception:
            pass
        print(f"✅ 检测到 GPU: {gpus}，已启用混合精度与显存按需分配。")
    else:
        print("⚠️ 未检测到 GPU，将使用 CPU 运行。")
except Exception as e:
    print(f"❌ GPU 配置错误: {e}")


# ====================================================

def ensure_dirs(base_dir: str):
    results_dir = os.path.join(base_dir, "results")
    logs_dir = os.path.join(base_dir, "logs")
    for sub in ["ib", "noib"]:
        os.makedirs(os.path.join(results_dir, sub), exist_ok=True)
        os.makedirs(os.path.join(logs_dir, sub), exist_ok=True)
    return results_dir, logs_dir


class TargetAccCallback(keras.callbacks.Callback):
    def __init__(self, target_acc):
        super().__init__()
        self.target_acc = target_acc

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        class_output_loss = logs.get("class_output_loss")
        if class_output_loss is not None and class_output_loss <= self.target_acc:
            print(f"\nReached target loss value {self.target_acc:.4f}; cancelling training!\n")
            self.model.stop_training = True


class BetaScheduler(keras.callbacks.Callback):
    """用于动态调整 IB 的 Beta 值"""

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
            # 余弦预热
            beta_now = self.beta_end - (self.beta_end - self.beta_start) * 0.5 * (1.0 + math.cos(math.pi * progress))

        for layer in self.model.layers:
            if hasattr(layer, "beta"):
                try:
                    layer.beta = float(beta_now)
                except Exception:
                    pass


def run_one_subject(subject_id: int, use_ib: bool, project_root: str, args: argparse.Namespace):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = script_dir
    results_dir, logs_dir = ensure_dirs(base_dir)

    subject_path = os.path.join(project_root, "data", "model_input")
    subject_file = f"subject_{subject_id:02d}_MI.npz"
    subject_filepath = os.path.join(subject_path, subject_file)

    if not os.path.exists(subject_filepath):
        raise FileNotFoundError(f"目标数据文件不存在: {subject_filepath}")

    with np.load(subject_filepath) as data:
        eeg = data["eeg"]
        fnirs = data["fnirs"]
        label = data["label"]

    fnirs *= 1e3
    label = label.astype(float)

    n_samples = eeg.shape[0]
    if n_samples < 600:
        raise ValueError(f"{subject_file} 仅包含 {n_samples} 个样本，无法执行 3 折交叉验证。")

    fold_results = []

    for session in range(3):
        print(f"--- Session {session} [Subject {subject_id:02d}] ---")

        # 1. CPU 端数据切分
        session_slice = slice(session * 200, (session + 1) * 200)

        # 剩余数据 (Train 2nd Stage)
        all_eeg = np.delete(eeg, session_slice, axis=0)
        all_fnirs = np.delete(fnirs, session_slice, axis=0)
        all_label = np.delete(label, session_slice, axis=0)

        # 测试数据
        eeg_test = eeg[session_slice]
        fnirs_test = fnirs[session_slice]
        label_test = label[session_slice]

        # 训练/验证划分 (Few-shot Split)
        np.random.seed(args.seed)
        indices = np.random.choice(all_eeg.shape[0], size=80, replace=False)

        eeg_train = np.delete(all_eeg, indices, axis=0)
        fnirs_train = np.delete(all_fnirs, indices, axis=0)
        label_train = np.delete(all_label, indices, axis=0)

        eeg_val = all_eeg[indices]
        fnirs_val = all_fnirs[indices]
        label_val = all_label[indices]

        # 2. GPU 端数据上传 (核弹级优化)
        # 直接转为 Tensor 并锁在显存中，跳过 Dataset 管道
        try:
            with tf.device('/GPU:0'):
                # First Stage Train Data
                x_train = {"eeg_input": tf.constant(eeg_train, tf.float32),
                           "fnirs_input": tf.constant(fnirs_train, tf.float32)}
                y_train = {"class_output": tf.constant(label_train, tf.float32),
                           "eeg_output": tf.constant(label_train, tf.float32)}

                # Validation Data
                x_val = {"eeg_input": tf.constant(eeg_val, tf.float32),
                         "fnirs_input": tf.constant(fnirs_val, tf.float32)}
                y_val = {"class_output": tf.constant(label_val, tf.float32),
                         "eeg_output": tf.constant(label_val, tf.float32)}

                # Second Stage Train Data (All remaining)
                x_train2 = {"eeg_input": tf.constant(all_eeg, tf.float32),
                            "fnirs_input": tf.constant(all_fnirs, tf.float32)}
                y_train2 = {"class_output": tf.constant(all_label, tf.float32),
                            "eeg_output": tf.constant(all_label, tf.float32)}

                # Test Data
                x_test = {"eeg_input": tf.constant(eeg_test, tf.float32),
                          "fnirs_input": tf.constant(fnirs_test, tf.float32)}
                y_test = {"class_output": tf.constant(label_test, tf.float32),
                          "eeg_output": tf.constant(label_test, tf.float32)}
        except RuntimeError as e:
            print(f"❌ 显存不足，无法一次性加载: {e}")
            raise e

        # 3. 模型构建
        tf.keras.backend.clear_session()
        if use_ib:
            model = sta_net_ib(latent_dim=args.latent_dim, beta=args.beta)
        else:
            model = sta_net()

        model.compile(
            optimizer="adam",
            # 关键优化：减少 CPU-GPU 通信
            steps_per_execution=10,
            # 关键修正：关闭 JIT 防止 libdevice 错误
            jit_compile=False,
            loss={
                "class_output": tf.keras.losses.SparseCategoricalCrossentropy(),
                "eeg_output": tf.keras.losses.SparseCategoricalCrossentropy(),
            },
            metrics={
                "class_output": ["accuracy"],
                "eeg_output": ["accuracy"],
            },
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_class_output_loss", patience=args.patience, restore_best_weights=True, verbose=0,
                mode="min"
            )
        ]

        # IB 模式特有的回调
        if use_ib:
            beta_sched = BetaScheduler(args.beta_start, args.beta, args.beta_warmup_epochs)
            callbacks.append(beta_sched)

        # 4. First Stage Training
        # verbose=0 静默模式，避免刷屏，只在最后输出结果
        first_history = model.fit(
            x_train, y_train,
            batch_size=args.batch_size,
            epochs=args.first_stage_epochs,
            verbose=0,
            validation_data=(x_val, y_val),
            callbacks=callbacks,
            shuffle=True
        )

        # 获取最佳 loss 用于第二阶段停止
        min_loss = min(first_history.history["val_class_output_loss"])
        min_epoch = first_history.history["val_class_output_loss"].index(min_loss)
        target_acc_val = first_history.history["class_output_loss"][min_epoch]

        # 5. Second Stage Training
        callbacks_stage2 = [TargetAccCallback(target_acc_val)]
        if use_ib:
            # IB 模式下保持 beta 更新（虽然此时应该已经到达 target beta）
            callbacks_stage2.append(beta_sched)

        model.fit(
            x_train2, y_train2,
            batch_size=args.batch_size,
            epochs=args.second_stage_epochs,
            verbose=0,
            callbacks=callbacks_stage2,
            shuffle=True
        )

        # 6. Test
        test_results = model.evaluate(x_test, y_test, batch_size=args.batch_size, verbose=0)
        # test_results: [total_loss, class_loss, eeg_loss, class_acc, eeg_acc, ...]
        print(f"  -> Result: Acc {test_results[3]:.4f} | Loss {test_results[1]:.4f}")

        fold_results.append(test_results)

        # 7. 内存清理 (非常重要，防止批量跑时 OOM)
        del x_train, y_train, x_val, y_val, x_train2, y_train2, x_test, y_test
        del model
        gc.collect()

    # 汇总结果
    min_len = min(len(fr) for fr in fold_results)
    fold_results = [fr[:min_len] for fr in fold_results]
    mean_results = np.mean(np.array(fold_results), axis=0).tolist()

    variant = "ib" if use_ib else "noib"
    out_path = os.path.join(results_dir, variant, f"subject_{subject_id:02d}_MI.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "subject": subject_id,
                "variant": variant,
                "fold_results": fold_results,
                "mean_results": mean_results,
                "params": vars(args)  # 记录实验参数
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return out_path, mean_results


def parse_args():
    parser = argparse.ArgumentParser(description="Run STA-Net experiments over multiple subjects.")
    parser.add_argument("--start", type=int, default=1, help="Start subject id.")
    parser.add_argument("--end", type=int, default=29, help="End subject id.")

    # 优化后的默认参数 (适配 RTX 4070)
    parser.add_argument("--batch-size", type=int, default=128, help="Increased batch size for GPU efficiency")

    parser.add_argument("--first-stage-epochs", type=int, default=300)
    parser.add_argument("--second-stage-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    # IB Hyperparams (最佳实践值)
    # Beta 2e-4 是我们刚才测出来的最佳值
    parser.add_argument("--beta", type=float, default=2e-4)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-warmup-epochs", type=int, default=30)

    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--only", choices=["ib", "noib", "both"], default="both")
    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    ensure_dirs(script_dir)

    summary = []

    print(f"🚀 批量训练任务启动! Subject {args.start} -> {args.end}")
    print(f"📦 参数配置: Batch={args.batch_size}, Beta={args.beta}, Latent={args.latent_dim}")

    for sid in range(args.start, args.end + 1):
        print(f"\n{'=' * 40}")
        print(f"处理被试: Subject {sid:02d}")
        print(f"{'=' * 40}")

        # 跑 NoIB (Standard)
        if args.only in ("both", "noib"):
            try:
                out_noib, mean_noib = run_one_subject(sid, use_ib=False, project_root=project_root, args=args)
                print(f"✅ [NoIB] 完成! Mean Acc: {mean_noib[3]:.4f} | Saved: {out_noib}")
                summary.append({"subject": sid, "variant": "noib", "mean_results": mean_noib})
            except Exception as e:
                print(f"❌ [NoIB] 失败: {e}")

        # 跑 IB
        if args.only in ("both", "ib"):
            try:
                out_ib, mean_ib = run_one_subject(sid, use_ib=True, project_root=project_root, args=args)
                print(f"✅ [IB]   完成! Mean Acc: {mean_ib[3]:.4f} | Saved: {out_ib}")
                summary.append({"subject": sid, "variant": "ib", "mean_results": mean_ib})
            except Exception as e:
                print(f"❌ [IB]   失败: {e}")

    # 保存总表
    all_summary_path = os.path.join(script_dir, "results", "summary_all.json")
    with open(all_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n🎉 所有任务全部完成! 汇总结果已保存至: {all_summary_path}")


if __name__ == "__main__":
    main()