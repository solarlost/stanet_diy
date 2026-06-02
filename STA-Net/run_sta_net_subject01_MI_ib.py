
from sta import sta_net_ib

import argparse
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
import math
import gc  # 引入垃圾回收


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train STA-Net with Information Bottleneck on subject_01 MI dataset."
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1e-4,
        help="Information bottleneck strength (KL multiplier).",
    )
    parser.add_argument(
        "--beta-start",
        type=float,
        default=1e-5,
        help="Warm-up start value for IB beta.",
    )
    parser.add_argument(
        "--beta-end",
        type=float,
        default=1e-3,
        help="Warm-up end value for IB beta.",
    )
    parser.add_argument(
        "--beta-warmup-epochs",
        type=int,
        default=30,
        help="Number of epochs to warm up beta to end value.",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=128,
        help="Latent dimension for the information bottleneck layers.",
    )
    # 针对 RTX 4070，默认 batch size 调大到 128
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for training and evaluation.",
    )
    parser.add_argument(
        "--first-stage-epochs",
        type=int,
        default=300,
        help="Maximum epochs for the first-stage (few-shot) training.",
    )
    parser.add_argument(
        "--second-stage-epochs",
        type=int,
        default=200,
        help="Maximum epochs for the second-stage fine-tuning.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early stopping patience (measured on validation loss).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility when splitting folds.",
    )
    return parser.parse_args()


class TargetAccCallback(keras.callbacks.Callback):
    def __init__(self, target_acc):
        super().__init__()
        self.target_acc = target_acc

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        class_output_loss = logs.get("class_output_loss")
        if class_output_loss is not None and class_output_loss <= self.target_acc:
            print(
                f"\nReached target loss value {self.target_acc:.4f}; cancelling second-stage early!\n"
            )
            self.model.stop_training = True


class BetaScheduler(keras.callbacks.Callback):
    def __init__(self, beta_start: float, beta_end: float, warmup_epochs: int):
        super().__init__()
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.warmup_epochs = max(1, int(warmup_epochs))

    def on_epoch_begin(self, epoch, logs=None):
        if epoch >= self.warmup_epochs:
            beta_now = self.beta_end
        else:
            progress = epoch / float(self.warmup_epochs)
            # cosine warm-up
            beta_now = self.beta_end - (self.beta_end - self.beta_start) * 0.5 * (1.0 + math.cos(math.pi * progress))
        # update beta on all layers that have a beta attribute (IB layers)
        for layer in self.model.layers:
            if hasattr(layer, "beta"):
                try:
                    layer.beta = float(beta_now)
                except Exception:
                    pass


def main():
    args = parse_args()

    # ==================== GPU 配置 ====================
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            # 开启混合精度加速 (RTX 4070 强项)
            try:
                '''tf.keras.mixed_precision.set_global_policy('mixed_float16')
                print("✅ 已启用混合精度 (Mixed Precision)")'''
                pass
            except Exception:
                pass

            print(f"✅ 检测到 GPU: {gpus}，已优化配置。")
        else:
            print("⚠️ 未检测到 GPU，使用 CPU。")
    except Exception as e:
        print(f"❌ GPU 配置错误: {e}")
    # ================================================

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    subject_path = os.path.join(project_root, "data", "model_input")
    subject_file = "subject_01_MI.npz"
    subject_filepath = os.path.join(subject_path, subject_file)

    if not os.path.exists(subject_filepath):
        raise FileNotFoundError(
            f"目标数据文件不存在: {subject_filepath}\n请确认已经完成预处理并生成该文件。"
        )

    print("开始训练第一个被试（MI）数据集 - 信息瓶颈版本 (GPU 极速版)")
    print(f"加载文件: {subject_file}")
    print(f"信息瓶颈超参数 beta={args.beta}, latent_dim={args.latent_dim}")

    with np.load(subject_filepath) as data:
        eeg = data["eeg"]
        fnirs = data["fnirs"]
        label = data["label"]

    fnirs *= 1e3
    label = label.astype(np.int32)

    n_samples = eeg.shape[0]
    if n_samples < 600:
        raise ValueError(
            f"{subject_file} 仅包含 {n_samples} 个样本，无法执行 3 折交叉验证。"
        )

    for session in range(3):
        print(f"\n{'=' * 20} 被试: subject_01_MI, 会话: {session} {'=' * 20}")

        # --- 1. 数据切分 (CPU) ---
        session_slice = slice(session * 200, (session + 1) * 200)

        # 所有剩余数据 (用于 Second Train)
        all_eeg = np.delete(eeg, session_slice, axis=0)
        all_fnirs = np.delete(fnirs, session_slice, axis=0)
        all_label = np.delete(label, session_slice, axis=0)

        # 测试数据
        eeg_test = eeg[session_slice]
        fnirs_test = fnirs[session_slice]
        label_test = label[session_slice]

        # 训练/验证划分
        np.random.seed(args.seed)
        indices = np.random.choice(all_eeg.shape[0], size=80, replace=False)

        eeg_train = np.delete(all_eeg, indices, axis=0)
        fnirs_train = np.delete(all_fnirs, indices, axis=0)
        label_train = np.delete(all_label, indices, axis=0)

        eeg_val = all_eeg[indices]
        fnirs_val = all_fnirs[indices]
        label_val = all_label[indices]

        if session == 0:
            print(f"数据形状: Train EEG {eeg_train.shape}, Val EEG {eeg_val.shape}")

        # --- 2. 核弹级优化：直接上传显存 (GPU Tensor) ---
        print("正在将数据上传至 GPU 显存...")
        try:
            with tf.device('/GPU:0'):
                # 训练集 1
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

                # 训练集 2 (Second Stage)
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
            print(f"❌ 显存不足或错误: {e}")
            return

        # --- 3. 模型编译 ---
        tf.keras.backend.clear_session()
        model = sta_net_ib(latent_dim=args.latent_dim, beta=args.beta)

        model.compile(
            optimizer="adam",
            # 关键参数：让 GPU 一口气跑完 10 步再回调，极大减少 Python 开销
            steps_per_execution=10,
            # 关键参数：关闭 JIT，防止 libdevice 缺失报错
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

        stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_class_output_loss",
            patience=args.patience,
            restore_best_weights=True,
            verbose=1,
            mode="min",
        )
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_class_output_loss",
            factor=0.5,
            patience=max(3, args.patience // 4),
            min_lr=1e-6,
            verbose=1,
            mode="min",
        )
        beta_sched = BetaScheduler(args.beta_start, args.beta_end, args.beta_warmup_epochs)

        # --- 4. 训练 First Stage ---
        print("begin first train (information bottleneck)")
        first_history = model.fit(
            x_train_gpu, y_train_gpu,  # 直接传 GPU Tensor
            batch_size=args.batch_size,
            epochs=args.first_stage_epochs,
            verbose=2,
            validation_data=(x_val_gpu, y_val_gpu),
            callbacks=[stopping, beta_sched, reduce_lr],
            shuffle=True
        )

        min_val_class_output_loss = min(first_history.history["val_class_output_loss"])
        min_val_class_output_loss_epoch = first_history.history[
            "val_class_output_loss"
        ].index(min_val_class_output_loss)
        target_acc = first_history.history["class_output_loss"][
            min_val_class_output_loss_epoch
        ]

        # --- 5. 训练 Second Stage ---
        print("begin second train (information bottleneck)")
        model.fit(
            x_train2_gpu, y_train2_gpu,
            batch_size=args.batch_size,
            epochs=args.second_stage_epochs,
            verbose=2,
            callbacks=[TargetAccCallback(target_acc), beta_sched, reduce_lr],
            shuffle=True
        )

        # --- 6. 测试 ---
        print("begin test (information bottleneck)")
        test_results = model.evaluate(x_test_gpu, y_test_gpu, batch_size=args.batch_size)
        print(f"测试结果（信息瓶颈）: {test_results}")

        # --- 7. 清理显存 ---
        del x_train_gpu, y_train_gpu, x_val_gpu, y_val_gpu
        del x_train2_gpu, y_train2_gpu, x_test_gpu, y_test_gpu
        gc.collect()

        # ==================== 新增：自动保存结果到 CSV ====================
        import csv
        result_file = "experiment_results.csv"
        file_exists = os.path.isfile(result_file)

        with open(result_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            # 如果文件不存在，先写表头
            if not file_exists:
                writer.writerow(['Beta', 'Latent_Dim', 'Seed', 'Class_Acc', 'EEG_Acc', 'Class_Loss', 'Total_Loss'])

            # 获取测试结果的关键指标
            # test_results 的顺序通常是 [loss, class_loss, eeg_loss, class_acc, eeg_acc, ...]
            # 请根据你的控制台输出确认索引，通常 class_acc 是第 3 个(索引3)，class_loss 是第 1 个(索引1)
            # 你的输出: [Loss, Class_Loss, EEG_Loss, Class_Acc, EEG_Acc, ...]
            total_loss = test_results[0]
            class_loss = test_results[1]
            class_acc = test_results[3]  # 确认一下你的输出顺序，如果是第4个就是索引3
            eeg_acc = test_results[4]

            writer.writerow([args.beta, args.latent_dim, args.seed, class_acc, eeg_acc, class_loss, total_loss])
            print(f"✅ 结果已追加保存到 {result_file}")
        # ==============================================================

    print("\nsubject_01_MI 信息瓶颈版本训练完成！")

if __name__ == "__main__":
    main()
