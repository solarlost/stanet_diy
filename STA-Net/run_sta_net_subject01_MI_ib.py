from sta import sta_net_ib

import argparse
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
import math


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train STA-Net with Information Bottleneck on subject_01 MI dataset."
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="快速模式：减少 epoch 与耐心值、可能增大 batch，以缩短运行时间。",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1e-3,
        help="Information bottleneck strength (KL multiplier).",
    )
    parser.add_argument(
        "--beta-start",
        type=float,
        default=1e-4,
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
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

    # 快速模式：调整默认训练超参，显著减少运行时间
    if args.fast:
        # 若用户未显式修改这些参数，则在快速模式下降低训练轮次与耐心值
        if args.first_stage_epochs == 300:
            args.first_stage_epochs = 120
        if args.second_stage_epochs == 200:
            args.second_stage_epochs = 80
        if args.patience == 50:
            args.patience = 12
        # 适度增大 batch（仅当用户未改动默认值）
        if args.batch_size == 32:
            args.batch_size = 48
        print(f"[FAST] 启用快速模式: first={args.first_stage_epochs}, second={args.second_stage_epochs}, patience={args.patience}, batch={args.batch_size}")

    # GPU setup: enable GPU if available and set memory growth and mixed precision
    try:
        gpus = tf.config.list_physical_devices('GPU')
        using_gpu = bool(gpus)
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except Exception:
                    pass
            # 启用 XLA（可能在某些模型上带来可观提速）
            try:
                tf.config.optimizer.set_jit(True)
            except Exception:
                pass
            try:
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
            except Exception:
                pass
            try:
                gpu_names = [tf.config.experimental.get_device_details(g).get('device_name', 'GPU') for g in gpus]
            except Exception:
                gpu_names = ['GPU' for _ in gpus]
            print(f"Detected GPU(s): {gpu_names}. Using GPU with mixed precision.")
        else:
            using_gpu = False
            print("No GPU detected. Training on CPU.")
    except Exception as e:
        using_gpu = False
        print(f"GPU configuration skipped due to error: {e}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    subject_path = os.path.join(project_root, "data", "model_input")
    subject_file = "subject_01_MI.npz"
    subject_filepath = os.path.join(subject_path, subject_file)

    if not os.path.exists(subject_filepath):
        raise FileNotFoundError(
            f"目标数据文件不存在: {subject_filepath}\n请确认已经完成预处理并生成该文件。"
        )

    print("开始训练第一个被试（MI）数据集 - 信息瓶颈版本")
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
        session_slice = slice(session * 200, (session + 1) * 200)
        all_eeg = np.delete(eeg, session_slice, axis=0)
        all_fnirs = np.delete(fnirs, session_slice, axis=0)
        all_label = np.delete(label, session_slice, axis=0)

        # ------- 数据集构建与优化：cache + prefetch -------
        options = tf.data.Options()
        options.experimental_optimization.apply_default_optimizations = True

        second_train_dataset = tf.data.Dataset.from_tensor_slices(
            (
                {"eeg_input": all_eeg, "fnirs_input": all_fnirs},
                {"class_output": all_label, "eeg_output": all_label},
            )
        )
        second_train_dataset = (
            second_train_dataset
            .shuffle(buffer_size=128)
            .batch(args.batch_size)
            .cache()
            .prefetch(tf.data.AUTOTUNE)
        ).with_options(options)

        eeg_test = eeg[session_slice]
        fnirs_test = fnirs[session_slice]
        label_test = label[session_slice]

        test_dataset = tf.data.Dataset.from_tensor_slices(
            (
                {"eeg_input": eeg_test, "fnirs_input": fnirs_test},
                {"class_output": label_test, "eeg_output": label_test},
            )
        ).batch(args.batch_size).cache().prefetch(tf.data.AUTOTUNE).with_options(options)

        np.random.seed(args.seed)
        indices = np.random.choice(all_eeg.shape[0], size=80, replace=False)

        eeg_train = np.delete(all_eeg, indices, axis=0)
        fnirs_train = np.delete(all_fnirs, indices, axis=0)
        label_train = np.delete(all_label, indices, axis=0)

        first_train_dataset = tf.data.Dataset.from_tensor_slices(
            (
                {"eeg_input": eeg_train, "fnirs_input": fnirs_train},
                {"class_output": label_train, "eeg_output": label_train},
            )
        ).shuffle(buffer_size=128).batch(args.batch_size).cache().prefetch(tf.data.AUTOTUNE).with_options(options)

        eeg_val = all_eeg[indices]
        fnirs_val = all_fnirs[indices]
        label_val = all_label[indices]
        val_dataset = tf.data.Dataset.from_tensor_slices(
            (
                {"eeg_input": eeg_val, "fnirs_input": fnirs_val},
                {"class_output": label_val, "eeg_output": label_val},
            )
        ).batch(args.batch_size).cache().prefetch(tf.data.AUTOTUNE).with_options(options)

        print("eeg_train shape:", eeg_train.shape)
        print("fnirs_train shape:", fnirs_train.shape)
        print("label_train shape:", label_train.shape)
        print("eeg_val shape:", eeg_val.shape)
        print("fnirs_val shape:", fnirs_val.shape)
        print("label_val shape:", label_val.shape)
        print(f"被试: subject_01_MI, 会话: {session}")

        tf.keras.backend.clear_session()
        model = sta_net_ib(latent_dim=args.latent_dim, beta=args.beta)

        # 在 GPU 上使用更大的 steps_per_execution；CPU 上保持为 1 以避免迭代器错误
        steps_per_exec = 50 if using_gpu else 1

        model.compile(
            optimizer="adam",
            # steps_per_execution: 减少 Python 与 TF 之间的调度开销
            steps_per_execution=steps_per_exec,
            loss={
                "class_output": tf.keras.losses.SparseCategoricalCrossentropy(),
                "eeg_output": tf.keras.losses.SparseCategoricalCrossentropy(),
            },
            metrics={
                "class_output": ["accuracy"],
                "eeg_output": ["accuracy"],
            },
            run_eagerly=False,
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

        print("begin first train (information bottleneck)")
        first_history = model.fit(
            first_train_dataset,
            epochs=args.first_stage_epochs,
            verbose=2,
            validation_data=val_dataset,
            callbacks=[stopping, beta_sched, reduce_lr],
        )

        min_val_class_output_loss = min(first_history.history["val_class_output_loss"])
        min_val_class_output_loss_epoch = first_history.history[
            "val_class_output_loss"
        ].index(min_val_class_output_loss)
        target_acc = first_history.history["class_output_loss"][
            min_val_class_output_loss_epoch
        ]

        print("begin second train (information bottleneck)")
        model.fit(
            second_train_dataset,
            epochs=args.second_stage_epochs,
            verbose=2,
            callbacks=[TargetAccCallback(target_acc), beta_sched, reduce_lr],
        )

        print("begin test (information bottleneck)")
        test_results = model.evaluate(test_dataset)
        print(f"测试结果（信息瓶颈）: {test_results}")

    print("\nsubject_01_MI 信息瓶颈版本训练完成！")


if __name__ == "__main__":
    main()




