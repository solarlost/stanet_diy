from sta import sta_net

import argparse
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train STA-Net (no IB) on subject_01 MI dataset using all sessions jointly."
    )
    parser.add_argument(
        "--metrics-out",
        type=str,
        default="",
        help="Optional path to write final evaluation metrics as JSON.",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=128,
        help="Latent dimension for the model, if applicable.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training and evaluation.",
    )
    parser.add_argument(
        "--few-shot-size",
        type=int,
        default=400,
        help="样本数量用于第一阶段（few-shot）训练，必须小于训练集大小。",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=80,
        help="验证/测试集大小，用于早停与最终评估。",
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


def build_dataset(eeg, fnirs, label, batch_size, shuffle=False, options=None):
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            {"eeg_input": eeg, "fnirs_input": fnirs},
            {"class_output": label, "eeg_output": label},
        )
    )
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(128, len(label)))
    dataset = dataset.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
    if options is not None:
        dataset = dataset.with_options(options)
    return dataset


def main():
    args = parse_args()

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

    print("开始训练第一个被试（MI）数据集 - 无信息瓶颈（全量训练）")
    print(f"加载文件: {subject_file}")

    with np.load(subject_filepath) as data:
        eeg = data["eeg"]
        fnirs = data["fnirs"]
        label = data["label"]

    fnirs *= 1e3
    label = label.astype(np.int32)

    n_samples = eeg.shape[0]
    if n_samples < args.val_size + 2:
        raise ValueError(
            f"{subject_file} 仅包含 {n_samples} 个样本，无法划分验证集。"
        )

    rng = np.random.default_rng(args.seed)
    all_indices = np.arange(n_samples)
    rng.shuffle(all_indices)

    val_size = min(args.val_size, n_samples // 5 if n_samples // 5 > 0 else args.val_size)
    val_indices = all_indices[:val_size]
    train_indices = all_indices[val_size:]

    if len(train_indices) <= 0:
        raise ValueError("训练集为空，请减少 --val-size 或检查数据集。")

    few_shot_size = min(args.few_shot_size, len(train_indices))
    few_shot_indices = train_indices[:few_shot_size]

    options = tf.data.Options()
    options.experimental_optimization.apply_default_optimizations = True

    first_train_dataset = build_dataset(
        eeg[few_shot_indices],
        fnirs[few_shot_indices],
        label[few_shot_indices],
        args.batch_size,
        shuffle=True,
        options=options,
    )

    full_train_dataset = build_dataset(
        eeg[train_indices],
        fnirs[train_indices],
        label[train_indices],
        args.batch_size,
        shuffle=True,
        options=options,
    )

    val_dataset = build_dataset(
        eeg[val_indices],
        fnirs[val_indices],
        label[val_indices],
        args.batch_size,
        shuffle=False,
        options=options,
    )

    print("few-shot train shape:", eeg[few_shot_indices].shape)
    print("full train shape:", eeg[train_indices].shape)
    print("val/test shape:", eeg[val_indices].shape)

    tf.keras.backend.clear_session()
    model = sta_net()

    steps_per_exec = 50 if using_gpu else 1

    model.compile(
        optimizer="adam",
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

    print("begin first train (few-shot)")
    first_history = model.fit(
        first_train_dataset,
        epochs=args.first_stage_epochs,
        verbose=2,
        validation_data=val_dataset,
        callbacks=[stopping, reduce_lr],
    )

    min_val_class_output_loss = min(first_history.history["val_class_output_loss"])
    min_val_class_output_loss_epoch = first_history.history[
        "val_class_output_loss"
    ].index(min_val_class_output_loss)
    target_acc = first_history.history["class_output_loss"][
        min_val_class_output_loss_epoch
    ]

    class TargetAccCallback(keras.callbacks.Callback):
        def __init__(self, target_acc_value):
            super().__init__()
            self.target_acc_value = target_acc_value

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            class_output_loss = logs.get("class_output_loss")
            if class_output_loss is not None and class_output_loss <= self.target_acc_value:
                print(
                    f"\nReached target loss value {self.target_acc_value:.4f}; cancelling second-stage early!\n"
                )
                self.model.stop_training = True

    print("begin second train (full data)")
    model.fit(
        full_train_dataset,
        epochs=args.second_stage_epochs,
        verbose=2,
        validation_data=val_dataset,
        callbacks=[TargetAccCallback(target_acc), reduce_lr],
    )

    print("begin evaluation on held-out set")
    test_results = model.evaluate(val_dataset, verbose=0)
    # Map metrics to names for robust downstream parsing
    metric_names = model.metrics_names
    metrics = {name: float(value) for name, value in zip(metric_names, test_results)}
    print(f"验证/测试结果（无信息瓶颈，全量训练）: {metrics}")
    # Emit a stable parseable line and optionally write to file
    try:
        import json as _json
        print("METRICS_JSON:", _json.dumps(metrics, ensure_ascii=False))
        if args.metrics_out:
            with open(args.metrics_out, "w", encoding="utf-8") as f:
                _json.dump(metrics, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    print("\nsubject_01_MI 无信息瓶颈全量训练完成！")


if __name__ == "__main__":
    main()


