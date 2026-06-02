import subprocess
import os
import shutil
import json
import sys
import time

def run_command(cmd):
    print(f"执行命令: {cmd}")
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        print(f"⚠️ 命令执行返回非零代码: {ret}")
        return False
    return True

def main():
    # ================= 配置区域 =================
    # 最佳 Beta 参数
    BEST_BETA = 5e-4
    
    # 脚本所在目录 (D:/test/stanet_diy/1223)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 结果存放目录 (D:/test/stanet_diy/1223/5e-4)
    output_dir = os.path.join(current_dir, "5e-4")
    os.makedirs(output_dir, exist_ok=True)
    
    # 项目根目录
    project_root = os.path.dirname(current_dir)
    
    # 训练脚本路径
    script_path = os.path.join(project_root, "STA-Net", "all.py")
    
    # 原始结果生成目录
    default_results_dir = os.path.join(project_root, "STA-Net", "results")
    
    # Python 解释器
    python_exe = sys.executable
    
    summary = []
    
    print(f"🚀 全员训练任务启动 (Subject 1 - 29)")
    print(f"🎯 目标 Beta: {BEST_BETA}")
    print(f"📂 结果保存至: {output_dir}")
    print("-" * 60)

    # 遍历所有被试
    for subject_id in range(1, 30):
        print(f"\n{'='*20} 处理 Subject {subject_id:02d} {'='*20}")
        
        # ---------------- 1. 跑 NoIB (基准) ----------------
        print(f"  >>> [NoIB] Running...")
        cmd_noib = f'"{python_exe}" "{script_path}" --start {subject_id} --end {subject_id} --only noib'
        
        noib_acc = 0.0
        noib_loss = 0.0
        
        if run_command(cmd_noib):
            source_file = os.path.join(default_results_dir, "noib", f"subject_{subject_id:02d}_MI.json")
            target_file = os.path.join(output_dir, f"subject_{subject_id:02d}_noib.json")
            
            if os.path.exists(source_file):
                with open(source_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    noib_acc = data['mean_results'][3]
                    noib_loss = data['mean_results'][1]
                
                if os.path.exists(target_file):
                    os.remove(target_file)
                shutil.move(source_file, target_file)
                print(f"  ✅ [NoIB] Done! Acc: {noib_acc:.4f}")
            else:
                print(f"  ❌ [NoIB] Result file not found.")
        else:
            print(f"  ❌ [NoIB] Training failed.")

        # ---------------- 2. 跑 IB (最佳参数) ----------------
        print(f"  >>> [IB]   Running (Beta={BEST_BETA})...")
        cmd_ib = f'"{python_exe}" "{script_path}" --start {subject_id} --end {subject_id} --only ib --beta {BEST_BETA}'
        
        ib_acc = 0.0
        ib_loss = 0.0
        
        if run_command(cmd_ib):
            source_file = os.path.join(default_results_dir, "ib", f"subject_{subject_id:02d}_MI.json")
            target_file = os.path.join(output_dir, f"subject_{subject_id:02d}_ib.json")
            
            if os.path.exists(source_file):
                with open(source_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    ib_acc = data['mean_results'][3]
                    ib_loss = data['mean_results'][1]
                
                if os.path.exists(target_file):
                    os.remove(target_file)
                shutil.move(source_file, target_file)
                print(f"  ✅ [IB]   Done! Acc: {ib_acc:.4f}")
            else:
                print(f"  ❌ [IB]   Result file not found.")
        else:
            print(f"  ❌ [IB]   Training failed.")
            
        # 记录该被试的对比结果
        diff = ib_acc - noib_acc
        summary.append({
            "subject": subject_id,
            "noib_acc": noib_acc,
            "ib_acc": ib_acc,
            "diff": diff,
            "noib_loss": noib_loss,
            "ib_loss": ib_loss
        })
        
        # 实时保存/更新汇总 CSV (防止中途断电数据丢失)
        summary_csv = os.path.join(output_dir, "summary_all_subjects.csv")
        with open(summary_csv, "w", encoding="utf-8") as f:
            f.write("subject,noib_acc,ib_acc,diff,noib_loss,ib_loss\n")
            for item in summary:
                f.write(f"{item['subject']},{item['noib_acc']:.6f},{item['ib_acc']:.6f},{item['diff']:.6f},{item['noib_loss']:.6f},{item['ib_loss']:.6f}\n")

    # ================= 最终报告 =================
    print("\n" + "=" * 60)
    print("📊 全员测试完成! 最终汇总")
    print("=" * 60)
    print(f"{'Sub':<5} | {'NoIB Acc':<10} | {'IB Acc':<10} | {'Diff':<10}")
    print("-" * 50)
    
    avg_noib = sum(x['noib_acc'] for x in summary) / len(summary) if summary else 0
    avg_ib = sum(x['ib_acc'] for x in summary) / len(summary) if summary else 0
    
    for item in summary:
        diff_str = f"{item['diff']:+.4f}"
        print(f"{item['subject']:<5} | {item['noib_acc']:.4f}     | {item['ib_acc']:.4f}     | {diff_str}")
        
    print("-" * 50)
    print(f"AVERAGE | {avg_noib:.4f}     | {avg_ib:.4f}     | {avg_ib - avg_noib:+.4f}")
    print(f"\n详细结果已保存至: {os.path.join(output_dir, 'summary_all_subjects.csv')}")

if __name__ == "__main__":
    main()