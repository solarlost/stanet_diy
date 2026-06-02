import subprocess
import os
import shutil
import json
import sys

def run_command(cmd):
    print(f"执行命令: {cmd}")
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        print(f"⚠️ 命令执行返回非零代码: {ret}")
        # 不抛出异常，以便继续执行后续测试
        return False
    return True

def main():
    # ================= 配置区域 =================
    # 当前脚本所在目录 (D:/test/stanet_diy/1223)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 项目根目录 (D:/test/stanet_diy)
    project_root = os.path.dirname(current_dir)
    
    # 训练脚本路径
    script_path = os.path.join(project_root, "STA-Net", "all.py")
    
    # 原始结果生成目录 (STA-Net/results)
    default_results_dir = os.path.join(project_root, "STA-Net", "results")
    
    # 待测试的 Beta 参数列表
    betas = [1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5]
    
    # Python 解释器
    python_exe = sys.executable
    
    summary = []
    
    print(f"🚀 开始 Subject 01 参数扫描测试")
    print(f"📂 结果将保存在: {current_dir}")
    print("-" * 50)

    # ================= 第一步: 测试 NoIB (基准) =================
    print(f"\n>>> [1/2] 正在测试 NoIB (Standard) 模式...")
    
    # NoIB 不受 Beta 影响，跑一次即可
    cmd_noib = f'"{python_exe}" "{script_path}" --start 1 --end 1 --only noib'
    
    if run_command(cmd_noib):
        source_file = os.path.join(default_results_dir, "noib", "subject_01_MI.json")
        target_file = os.path.join(current_dir, "subject_01_noib.json")
        
        if os.path.exists(source_file):
            with open(source_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                acc = data['mean_results'][3]
                loss = data['mean_results'][1]
            
            # 移动并重命名文件
            if os.path.exists(target_file):
                os.remove(target_file)
            shutil.move(source_file, target_file)
            
            print(f"✅ NoIB: Acc = {acc:.4f}, Loss = {loss:.4f}")
            summary.append({
                "type": "noib",
                "beta": "N/A",
                "acc": acc,
                "loss": loss,
                "file": os.path.basename(target_file)
            })
        else:
            print("❌ NoIB 结果文件未找到。")
    else:
        print("❌ NoIB 训练失败。")

    # ================= 第二步: 测试 IB (不同 Beta) =================
    print(f"\n>>> [2/2] 正在测试 IB 模式 (共 {len(betas)} 个参数)...")
    
    for beta in betas:
        print(f"\n   -> Testing Beta = {beta}")
        
        cmd_ib = f'"{python_exe}" "{script_path}" --start 1 --end 1 --only ib --beta {beta}'
        
        if run_command(cmd_ib):
            source_file = os.path.join(default_results_dir, "ib", "subject_01_MI.json")
            target_file = os.path.join(current_dir, f"subject_01_ib_beta_{beta}.json")
            
            if os.path.exists(source_file):
                with open(source_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    acc = data['mean_results'][3]
                    loss = data['mean_results'][1]
                
                if os.path.exists(target_file):
                    os.remove(target_file)
                shutil.move(source_file, target_file)
                
                print(f"   ✅ IB (Beta {beta}): Acc = {acc:.4f}, Loss = {loss:.4f}")
                summary.append({
                    "type": "ib",
                    "beta": beta,
                    "acc": acc,
                    "loss": loss,
                    "file": os.path.basename(target_file)
                })
            else:
                print(f"   ❌ IB (Beta {beta}) 结果文件未找到。")
        else:
            print(f"   ❌ IB (Beta {beta}) 训练失败。")

    # ================= 第三步: 汇总报告 =================
    print("\n" + "=" * 60)
    print("📊 测试汇总 (Subject 01)")
    print("=" * 60)
    print(f"{'Type':<10} | {'Beta':<10} | {'Acc':<10} | {'Loss':<10}")
    print("-" * 55)
    
    # 按准确率排序
    summary.sort(key=lambda x: x['acc'], reverse=True)
    
    for item in summary:
        beta_str = str(item['beta'])
        print(f"{item['type']:<10} | {beta_str:<10} | {item['acc']:.4f}     | {item['loss']:.4f}")
    
    # 保存 CSV 到 1223 文件夹
    summary_csv = os.path.join(current_dir, "summary_1223.csv")
    with open(summary_csv, "w", encoding="utf-8") as f:
        f.write("type,beta,acc,loss,file\n")
        for item in summary:
            f.write(f"{item['type']},{item['beta']},{item['acc']},{item['loss']},{item['file']}\n")
            
    print(f"\n详细结果已保存至: {summary_csv}")

if __name__ == "__main__":
    main()