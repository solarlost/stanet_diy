import subprocess
import os
import sys
import time
import json
import numpy as np
import pandas as pd

def run_command(cmd):
    print(f"执行命令: {cmd}")
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        print(f"⚠️ 命令执行返回非零代码: {ret}")
        return False
    return True

def main():
    # ================= 配置区域 =================
    REPEAT_TIMES = 3  # 每个被试重复跑 3 次取平均
    
    # EF-IB 参数 (稳健版)
    BETA = 1e-4
    ALIGN_WEIGHT = 0.001
    PATIENCE = 100
    
    # 路径设置
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    
    # 结果保存目录
    results_dir = os.path.join(current_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # 脚本路径
    # 我们复用之前的脚本，但需要确保路径引用正确
    # 1. NoIB 脚本: 使用 STA-Net/all.py
    script_noib = os.path.join(project_root, "STA-Net", "all.py")
    
    # 2. EF-IB 脚本: 使用 1223/改进模型架构/run_efib_sub01.py
    # 注意: 这个脚本原本只跑 Sub 01，我们需要稍微修改调用方式或者确保它能接受 --start --end 参数
    # 检查发现 run_efib_sub01.py 内部硬编码了 run_efib_subject(1, args)
    # 所以我们需要创建一个新的通用脚本 run_efib_generic.py 来支持任意被试
    script_efib = os.path.join(current_dir, "run_efib_generic.py")
    
    python_exe = sys.executable
    
    summary_data = []
    
    print(f"🚀 1224 全员严谨测试启动 (Subject 1-29, Repeat {REPEAT_TIMES})")
    print(f"⚙️ EF-IB 参数: Beta={BETA}, Align={ALIGN_WEIGHT}, Patience={PATIENCE}")
    print(f"📂 结果保存至: {results_dir}")
    print("-" * 60)

    for subject_id in range(1, 30):
        print(f"\n{'='*30}")
        print(f"处理 Subject {subject_id:02d}")
        print(f"{'='*30}")
        
        subject_results = {
            "subject": subject_id,
            "noib_accs": [],
            "efib_accs": [],
            "noib_mean": 0.0,
            "efib_mean": 0.0,
            "diff": 0.0
        }
        
        # ---------------- 1. 跑 NoIB (3次) ----------------
        print(f"\n>>> [NoIB] Running {REPEAT_TIMES} times...")
        for i in range(REPEAT_TIMES):
            seed = 42 + i
            print(f"   -> Run {i+1}/{REPEAT_TIMES} (Seed={seed})")
            
            # all.py 默认保存到 STA-Net/results/noib
            # 我们需要把它移动到 1224/results/noib
            cmd = f'"{python_exe}" "{script_noib}" --start {subject_id} --end {subject_id} --only noib --patience {PATIENCE} --seed {seed}'
            
            if run_command(cmd):
                # 读取结果
                src_file = os.path.join(project_root, "STA-Net", "results", "noib", f"subject_{subject_id:02d}_MI.json")
                if os.path.exists(src_file):
                    with open(src_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        acc = data['mean_results'][3]
                    subject_results["noib_accs"].append(acc)
                    print(f"      ✅ Acc: {acc:.4f}")
                    
                    # 备份文件
                    dst_dir = os.path.join(results_dir, "noib")
                    os.makedirs(dst_dir, exist_ok=True)
                    dst_file = os.path.join(dst_dir, f"subject_{subject_id:02d}_run{i+1}.json")
                    if os.path.exists(dst_file): os.remove(dst_file)
                    os.rename(src_file, dst_file)
                else:
                    print("      ❌ Result file not found.")
            else:
                print("      ❌ Training failed.")

        # ---------------- 2. 跑 EF-IB (3次) ----------------
        print(f"\n>>> [EF-IB] Running {REPEAT_TIMES} times...")
        for i in range(REPEAT_TIMES):
            seed = 42 + i
            print(f"   -> Run {i+1}/{REPEAT_TIMES} (Seed={seed})")
            
            # 调用我们将要创建的 run_efib_generic.py
            cmd = f'"{python_exe}" "{script_efib}" --subject {subject_id} --beta {BETA} --align-weight {ALIGN_WEIGHT} --patience {PATIENCE} --seed {seed}'
            
            if run_command(cmd):
                # 读取结果 (run_efib_generic.py 会生成到 1224/results/efib)
                src_file = os.path.join(results_dir, "efib", f"subject_{subject_id:02d}_efib.json")
                if os.path.exists(src_file):
                    with open(src_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        acc = data['mean_results'][3]
                    subject_results["efib_accs"].append(acc)
                    print(f"      ✅ Acc: {acc:.4f}")
                    
                    # 备份文件
                    dst_file = os.path.join(results_dir, "efib", f"subject_{subject_id:02d}_run{i+1}.json")
                    if os.path.exists(dst_file): os.remove(dst_file)
                    os.rename(src_file, dst_file)
                else:
                    print("      ❌ Result file not found.")
            else:
                print("      ❌ Training failed.")

        # ---------------- 3. 统计与保存 ----------------
        if subject_results["noib_accs"]:
            subject_results["noib_mean"] = np.mean(subject_results["noib_accs"])
        if subject_results["efib_accs"]:
            subject_results["efib_mean"] = np.mean(subject_results["efib_accs"])
            
        subject_results["diff"] = subject_results["efib_mean"] - subject_results["noib_mean"]
        
        summary_data.append(subject_results)
        
        # 实时保存汇总 CSV
        df = pd.DataFrame(summary_data)
        # 将列表转为字符串以便保存
        df['noib_accs'] = df['noib_accs'].apply(lambda x: str(x))
        df['efib_accs'] = df['efib_accs'].apply(lambda x: str(x))
        df.to_csv(os.path.join(results_dir, "summary_1224.csv"), index=False)
        
        print(f"\n📊 Subject {subject_id:02d} Summary:")
        print(f"   NoIB Mean: {subject_results['noib_mean']:.4f}")
        print(f"   EF-IB Mean: {subject_results['efib_mean']:.4f}")
        print(f"   Diff:       {subject_results['diff']:+.4f}")

    print("\n" + "=" * 60)
    print("🎉 所有任务完成!")
    print(f"结果已保存至: {os.path.join(results_dir, 'summary_1224.csv')}")

if __name__ == "__main__":
    main()