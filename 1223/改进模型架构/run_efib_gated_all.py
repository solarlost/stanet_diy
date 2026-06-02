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
    # 冠军参数
    BEST_BETA = 5e-4
    BEST_ALIGN_WEIGHT = 0.01
    
    # 脚本所在目录 (D:/test/stanet_diy/1223/改进模型架构)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 结果存放目录
    output_dir = os.path.join(current_dir, "results_gated_all")
    os.makedirs(output_dir, exist_ok=True)
    
    # 项目根目录
    project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
    
    # 训练脚本路径 (复用 run_efib_sub01.py，因为它已经封装好了 efib_net 的调用)
    # 我们只需要通过命令行传参来控制它跑哪个被试
    # 但 run_efib_sub01.py 目前是硬编码跑 Subject 1 的
    # 为了方便，我们直接在这里调用 run_efib_sub01.py 中的函数，而不是通过 subprocess
    # 这样更灵活
    
    # 动态导入 run_efib_sub01
    sys.path.append(current_dir)
    import run_efib_sub01
    
    # 原始 NoIB 结果目录 (用于对比)
    noib_results_dir = os.path.join(project_root, "STA-Net", "results", "noib")
    
    summary = []
    
    print(f"🚀 全员 Gated EF-IB Net 测试启动 (Subject 1 - 29)")
    print(f"🎯 参数: Beta={BEST_BETA}, Align={BEST_ALIGN_WEIGHT}")
    print(f"📂 结果保存至: {output_dir}")
    print("-" * 60)

    # 模拟 argparse 参数
    class Args:
        def __init__(self):
            self.batch_size = 128
            self.first_stage_epochs = 300
            self.second_stage_epochs = 200
            self.patience = 50
            self.seed = 42
            self.beta = BEST_BETA
            self.beta_start = 1e-4
            self.beta_warmup_epochs = 30
            self.latent_dim = 128
            self.align_weight = BEST_ALIGN_WEIGHT

    args = Args()

    for subject_id in range(1, 30):
        print(f"\n{'='*20} 处理 Subject {subject_id:02d} {'='*20}")
        
        # 1. 运行 Gated EF-IB
        try:
            # 修改 run_efib_sub01.py 中的 run_efib_subject 函数以支持自定义输出路径
            # 这里我们稍微 hack 一下，直接调用并移动结果
            
            # 调用训练函数
            # 注意：run_efib_sub01.py 里的 run_efib_subject 会在 current_dir/results 下生成文件
            temp_out_path, ib_acc = run_efib_sub01.run_efib_subject(subject_id, args)
            
            # 移动结果文件到我们的 output_dir
            final_out_path = os.path.join(output_dir, f"subject_{subject_id:02d}_gated_efib.json")
            if os.path.exists(final_out_path):
                os.remove(final_out_path)
            shutil.move(temp_out_path, final_out_path)
            
            print(f"  ✅ [Gated IB] Done! Acc: {ib_acc:.4f}")
            
            # 读取 Loss 信息 (从移动后的文件)
            with open(final_out_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                ib_loss = data['mean_results'][1] # class_loss
                
        except Exception as e:
            print(f"  ❌ [Gated IB] Failed: {e}")
            ib_acc = 0.0
            ib_loss = 0.0
            import traceback
            traceback.print_exc()

        # 2. 读取 NoIB 结果 (如果存在)
        noib_file = os.path.join(noib_results_dir, f"subject_{subject_id:02d}_MI.json")
        noib_acc = 0.0
        noib_loss = 0.0
        
        if os.path.exists(noib_file):
            try:
                with open(noib_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    noib_acc = data['mean_results'][3]
                    noib_loss = data['mean_results'][1]
                print(f"  ℹ️ [NoIB]     Found. Acc: {noib_acc:.4f}")
            except Exception:
                print(f"  ⚠️ [NoIB]     File corrupted.")
        else:
            print(f"  ⚠️ [NoIB]     File not found.")

        # 3. 记录对比
        diff = ib_acc - noib_acc
        summary.append({
            "subject": subject_id,
            "noib_acc": noib_acc,
            "ib_acc": ib_acc,
            "diff": diff,
            "noib_loss": noib_loss,
            "ib_loss": ib_loss
        })
        
        # 实时保存 CSV
        summary_csv = os.path.join(output_dir, "summary_gated_all.csv")
        with open(summary_csv, "w", encoding="utf-8") as f:
            f.write("subject,noib_acc,ib_acc,diff,noib_loss,ib_loss\n")
            for item in summary:
                f.write(f"{item['subject']},{item['noib_acc']:.6f},{item['ib_acc']:.6f},{item['diff']:.6f},{item['noib_loss']:.6f},{item['ib_loss']:.6f}\n")

    # ================= 最终报告 =================
    print("\n" + "=" * 60)
    print("📊 全员 Gated EF-IB 测试完成!")
    print("=" * 60)
    print(f"{'Sub':<5} | {'NoIB Acc':<10} | {'Gated Acc':<10} | {'Diff':<10}")
    print("-" * 50)
    
    avg_noib = sum(x['noib_acc'] for x in summary) / len(summary) if summary else 0
    avg_ib = sum(x['ib_acc'] for x in summary) / len(summary) if summary else 0
    
    for item in summary:
        diff_str = f"{item['diff']:+.4f}"
        print(f"{item['subject']:<5} | {item['noib_acc']:.4f}     | {item['ib_acc']:.4f}     | {diff_str}")
        
    print("-" * 50)
    print(f"AVERAGE | {avg_noib:.4f}     | {avg_ib:.4f}     | {avg_ib - avg_noib:+.4f}")
    print(f"\n详细结果已保存至: {os.path.join(output_dir, 'summary_gated_all.csv')}")

if __name__ == "__main__":
    main()