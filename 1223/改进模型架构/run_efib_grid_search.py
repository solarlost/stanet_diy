import subprocess
import os
import sys
import itertools
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
    # 待测试的参数网格
    betas = [5e-4, 2e-4, 1e-4]
    align_weights = [0.01, 0.005, 0.001]
    
    # 脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 训练脚本路径 (复用之前的 run_efib_sub01.py)
    script_path = os.path.join(current_dir, "run_efib_sub01.py")
    
    # 结果汇总文件
    summary_csv = os.path.join(current_dir, "summary_grid_search.csv")
    
    # Python 解释器
    python_exe = sys.executable
    
    print(f"🚀 启动 EF-IB Net 参数网格搜索")
    print(f"📍 待测 Betas: {betas}")
    print(f"📍 待测 Align Weights: {align_weights}")
    print(f"📂 结果汇总至: {summary_csv}")
    print("-" * 60)

    # 初始化 CSV (如果不存在)
    if not os.path.exists(summary_csv):
        with open(summary_csv, "w", encoding="utf-8") as f:
            f.write("beta,align_weight,acc,file\n")

    # 生成所有组合
    combinations = list(itertools.product(betas, align_weights))
    total_jobs = len(combinations)

    for idx, (beta, align_weight) in enumerate(combinations):
        print(f"\n[{idx+1}/{total_jobs}] 正在测试: Beta={beta}, Align={align_weight}")
        
        # 构造命令
        cmd = f'"{python_exe}" "{script_path}" --beta {beta} --align-weight {align_weight}'
        
        start_time = time.time()
        if run_command(cmd):
            # 运行成功后，读取生成的 json 结果
            # run_efib_sub01.py 默认生成 subject_01_efib.json
            result_file = os.path.join(current_dir, "results", "subject_01_efib.json")
            
            if os.path.exists(result_file):
                import json
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    acc = data['mean_results'][3] # class_acc
                
                print(f"✅ 任务完成! Acc: {acc:.4f}")
                
                # 重命名结果文件，防止覆盖
                new_filename = f"subject_01_efib_b{beta}_w{align_weight}.json"
                new_filepath = os.path.join(current_dir, "results", new_filename)
                
                if os.path.exists(new_filepath):
                    os.remove(new_filepath)
                os.rename(result_file, new_filepath)
                
                # 写入汇总 CSV
                with open(summary_csv, "a", encoding="utf-8") as f:
                    f.write(f"{beta},{align_weight},{acc:.6f},{new_filename}\n")
            else:
                print("❌ 结果文件未找到，可能训练未正常结束。")
        else:
            print("❌ 训练脚本执行失败。")
            
        print(f"⏱️ 耗时: {time.time() - start_time:.2f}s")

    print("\n" + "=" * 60)
    print("📊 网格搜索完成! 最终排行榜")
    print("=" * 60)
    
    if os.path.exists(summary_csv):
        import pandas as pd
        try:
            df = pd.read_csv(summary_csv)
            df = df.sort_values(by="acc", ascending=False)
            print(df.to_string(index=False))
        except Exception:
            # 如果没有 pandas，简单打印
            with open(summary_csv, 'r') as f:
                print(f.read())
    
    print(f"\n详细结果已保存至: {summary_csv}")

if __name__ == "__main__":
    main()