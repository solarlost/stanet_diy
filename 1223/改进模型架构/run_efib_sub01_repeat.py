import subprocess
import os
import sys
import numpy as np
import time
import json

def run_command(cmd):
    print(f"执行命令: {cmd}")
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        print(f"⚠️ 命令执行返回非零代码: {ret}")
        return False
    return True

def main():
    # ================= 配置区域 =================
    REPEAT_TIMES = 5  # 重复运行次数
    
    # 更保守的参数，旨在提高稳定性
    BETA = 1e-4          # 从 5e-4 降低
    ALIGN_WEIGHT = 0.001 # 从 0.01 降低
    PATIENCE = 100       # 从 50 增加，给模型更多收敛时间
    
    # 脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 训练脚本路径 (复用 run_efib_sub01.py)
    script_path = os.path.join(current_dir, "run_efib_sub01.py")
    
    # Python 解释器
    python_exe = sys.executable
    
    results = []
    
    print(f"🚀 启动 Subject 01 重复性测试 (稳健版)")
    print(f"⚙️ 参数: Beta={BETA}, Align={ALIGN_WEIGHT}, Patience={PATIENCE}")
    print("-" * 60)

    for i in range(REPEAT_TIMES):
        seed = 42 + i  # 每次使用不同的种子: 42, 43, 44, 45, 46
        print(f"\n🔄 [第 {i+1}/{REPEAT_TIMES} 次运行] Seed={seed}")
        
        # 构造命令
        cmd = f'"{python_exe}" "{script_path}" --beta {BETA} --align-weight {ALIGN_WEIGHT} --patience {PATIENCE} --seed {seed}'
        
        start_time = time.time()
        if run_command(cmd):
            # 读取结果
            result_file = os.path.join(current_dir, "results", "subject_01_efib.json")
            
            if os.path.exists(result_file):
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    acc = data['mean_results'][3]
                    loss = data['mean_results'][1]
                
                print(f"✅ 第 {i+1} 次完成! Acc: {acc:.4f}, Loss: {loss:.4f}")
                results.append(acc)
                
                # 备份结果文件
                new_filename = f"subject_01_efib_conservative_run{i+1}_seed{seed}.json"
                new_filepath = os.path.join(current_dir, "results", new_filename)
                if os.path.exists(new_filepath):
                    os.remove(new_filepath)
                os.rename(result_file, new_filepath)
            else:
                print("❌ 结果文件未找到。")
        else:
            print("❌ 训练脚本执行失败。")
            
        print(f"⏱️ 耗时: {time.time() - start_time:.2f}s")

    # ================= 统计分析 =================
    if results:
        mean_acc = np.mean(results)
        std_acc = np.std(results)
        max_acc = np.max(results)
        min_acc = np.min(results)
        
        print("\n" + "=" * 60)
        print("📊 重复测试统计报告 (Subject 01 - 稳健版)")
        print("=" * 60)
        print(f"运行次数: {len(results)}")
        print(f"平均准确率 (Mean): {mean_acc:.4f} ({mean_acc*100:.2f}%)")
        print(f"标准差 (Std):      {std_acc:.4f}")
        print(f"最高分 (Max):      {max_acc:.4f}")
        print(f"最低分 (Min):      {min_acc:.4f}")
        print("-" * 60)
        print(f"详细得分: {results}")
        
        # 保存报告
        report_file = os.path.join(current_dir, "summary_repeat_test_conservative.txt")
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(f"Mean: {mean_acc:.4f}\n")
            f.write(f"Std: {std_acc:.4f}\n")
            f.write(f"Results: {results}\n")
            f.write(f"Params: Beta={BETA}, Align={ALIGN_WEIGHT}, Patience={PATIENCE}\n")
        print(f"报告已保存至: {report_file}")
    else:
        print("\n❌ 没有收集到任何结果。")

if __name__ == "__main__":
    main()