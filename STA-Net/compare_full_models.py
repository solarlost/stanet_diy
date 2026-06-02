import json
import os
import subprocess
import sys
import tempfile


def run_script(script_path: str) -> dict:
    with tempfile.TemporaryDirectory() as td:
        out_path = os.path.join(td, "metrics.json")
        # Use unbuffered mode (-u) to stream logs in real-time
        cmd = [sys.executable, "-u", script_path, "--metrics-out", out_path]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"Command failed ({ret}): {' '.join(cmd)}")
        if not os.path.exists(out_path):
            raise FileNotFoundError(f"Metrics file not written by {script_path}")
        with open(out_path, "r", encoding="utf-8") as f:
            return json.load(f)


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    script_ib = os.path.join(root, "run_sta_net_subject01_MI_ib_full.py")
    script_noib = os.path.join(root, "run_sta_net_subject01_MI_full.py")

    print("==== Running IB full model ====")
    metrics_ib = run_script(script_ib)

    print("==== Running NO-IB full model ====")
    metrics_noib = run_script(script_noib)

    acc_key = "class_output_accuracy"
    ib_acc = metrics_ib.get(acc_key)
    noib_acc = metrics_noib.get(acc_key)

    print("\n==== Comparison (class_output_accuracy) ====")
    print(f"IB    {acc_key}: {ib_acc}")
    print(f"NO-IB {acc_key}: {noib_acc}")

    if ib_acc is not None and noib_acc is not None:
        better = "IB" if ib_acc > noib_acc else ("NO-IB" if noib_acc > ib_acc else "TIE")
        print(f"Winner: {better}")
    else:
        print("Warning: accuracy key not found in one or both metrics.")


if __name__ == "__main__":
    main()


