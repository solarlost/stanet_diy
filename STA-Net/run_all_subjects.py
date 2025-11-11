import os
import json
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras

from sta import sta_net, sta_net_ib


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


def run_one_subject(subject_id: int, use_ib: bool, project_root: str, args: argparse.Namespace):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = script_dir
    results_dir, logs_dir = ensure_dirs(base_dir)

    subject_path = os.path.join(project_root, "data", "model_input")
    subject_file = f"subject_{subject_id:02d}_MI.npz"
    subject_filepath = os.path.join(subject_path, subject_file)
    if not os.path.exists(subject_filepath):
        raise FileNotFoundError(
            f"目标数据文件不存在: {subject_filepath}\n请确认已经完成预处理并生成该文件。"
        )

    with np.load(subject_filepath) as data:
        eeg = data["eeg"]
        fnirs = data["fnirs"]
        label = data["label"]

    fnirs *= 1e3
    label = label.astype(float)

    n_samples = eeg.shape[0]
    if n_samples < 600:
        raise ValueError(
            f"{subject_file} 仅包含 {n_samples} 个样本，无法执行 3 折交叉验证。"
        )

    fold_results = []

    for session in range(3):
        session_slice = slice(session * 200, (session + 1) * 200)
        all_eeg = np.delete(eeg, session_slice, axis=0)
        all_fnirs = np.delete(fnirs, session_slice, axis=0)
        all_label = np.delete(label, session_slice, axis=0)

        # datasets
        second_train_dataset = tf.data.Dataset.from_tensor_slices(
            (
                {"eeg_input": all_eeg, "fnirs_input": all_fnirs},
                {"class_output": all_label, "eeg_output": all_label},
            )
        ).shuffle(buffer_size=128).batch(args.batch_size)

        eeg_test = eeg[session_slice]
        fnirs_test = fnirs[session_slice]
        label_test = label[session_slice]
        test_dataset = tf.data.Dataset.from_tensor_slices(
            (
                {"eeg_input": eeg_test, "fnirs_input": fnirs_test},
                {"class_output": label_test, "eeg_output": label_test},
            )
        ).batch(args.batch_size)

        # few-shot split
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
        ).shuffle(buffer_size=128).batch(args.batch_size)

        eeg_val = all_eeg[indices]
        fnirs_val = all_fnirs[indices]
        label_val = all_label[indices]
        val_dataset = tf.data.Dataset.from_tensor_slices(
            (
                {"eeg_input": eeg_val, "fnirs_input": fnirs_val},
                {"class_output": label_val, "eeg_output": label_val},
            )
        ).batch(args.batch_size)

        tf.keras.backend.clear_session()
        if use_ib:
            model = sta_net_ib(latent_dim=args.latent_dim, beta=args.beta)
        else:
            model = sta_net()

        model.compile(
            optimizer="adam",
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

        # first stage
        print(
            f"begin first train - subject_{subject_id:02d}_MI, session={session}, "
            f"{'IB' if use_ib else 'NoIB'}"
        )
        first_history = model.fit(
            first_train_dataset,
            epochs=args.first_stage_epochs,
            verbose=2,
            validation_data=val_dataset,
            callbacks=[stopping],
        )

        min_val_class_output_loss = min(first_history.history["val_class_output_loss"])
        min_val_epoch = first_history.history["val_class_output_loss"].index(
            min_val_class_output_loss
        )
        target_acc = first_history.history["class_output_loss"][min_val_epoch]

        # second stage
        print(
            f"begin second train - subject_{subject_id:02d}_MI, session={session}, "
            f"{'IB' if use_ib else 'NoIB'}"
        )
        model.fit(
            second_train_dataset,
            epochs=args.second_stage_epochs,
            verbose=2,
            callbacks=[TargetAccCallback(target_acc)],
        )

        # test
        print(
            f"begin test - subject_{subject_id:02d}_MI, session={session}, "
            f"{'IB' if use_ib else 'NoIB'}"
        )
        test_results = model.evaluate(test_dataset, verbose=2)
        # Keras returns: total_loss, class_output_loss, class_output_accuracy,
        #                eeg_output_loss, eeg_output_accuracy, ... (custom metrics may follow)
        fold_results.append(test_results)

    # aggregate (mean across 3 folds)
    # align length by min length (in case metric lists differ slightly)
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
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return out_path, mean_results


def parse_args():
    parser = argparse.ArgumentParser(description="Run STA-Net experiments over multiple subjects.")
    parser.add_argument("--start", type=int, default=1, help="Start subject id (inclusive).")
    parser.add_argument("--end", type=int, default=29, help="End subject id (inclusive).")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--first-stage-epochs", type=int, default=300)
    parser.add_argument("--second-stage-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    # IB hyperparams
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--only", choices=["ib", "noib", "both"], default="both")
    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    ensure_dirs(script_dir)

    summary = []
    for sid in range(args.start, args.end + 1):
        if args.only in ("both", "noib"):
            try:
                out_noib, mean_noib = run_one_subject(sid, use_ib=False, project_root=project_root, args=args)
                print(f"[NoIB] subject_{sid:02d} saved: {out_noib}")
                summary.append({"subject": sid, "variant": "noib", "mean_results": mean_noib})
            except Exception as e:
                print(f"[NoIB] subject_{sid:02d} failed: {e}")
        if args.only in ("both", "ib"):
            try:
                out_ib, mean_ib = run_one_subject(sid, use_ib=True, project_root=project_root, args=args)
                print(f"[IB] subject_{sid:02d} saved: {out_ib}")
                summary.append({"subject": sid, "variant": "ib", "mean_results": mean_ib})
            except Exception as e:
                print(f"[IB] subject_{sid:02d} failed: {e}")

    # Save all summary
    all_summary_path = os.path.join(script_dir, "results", "summary_all.json")
    with open(all_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"All results summary saved to: {all_summary_path}")


if __name__ == "__main__":
    main()




