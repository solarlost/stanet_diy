import pandas as pd
import os
from scipy import stats
import numpy as np

def main():
    # 路径设置
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_file = os.path.abspath(os.path.join(current_dir, "../1226/results/summary_ms_se_ib_vs_noib.csv"))
    
    print(f"📂 读取数据文件: {results_file}")
    
    if not os.path.exists(results_file):
        print("❌ 文件不存在！请先运行 1226/run_ms_se_all.py 生成结果。")
        return

    df = pd.read_csv(results_file)
    
    noib_accs = df['noib_acc'].values
    ms_se_ib_accs = df['ms_se_ib_acc'].values
    
    print(f"📊 样本数量: {len(df)}")
    print(f"   NoIB Mean:     {np.mean(noib_accs):.4f} ± {np.std(noib_accs):.4f}")
    print(f"   MS-SE-IB Mean: {np.mean(ms_se_ib_accs):.4f} ± {np.std(ms_se_ib_accs):.4f}")
    print("-" * 50)

    # 1. 配对 t 检验 (Paired t-test)
    # 假设差值服从正态分布
    t_stat, p_val_t = stats.ttest_rel(ms_se_ib_accs, noib_accs, alternative='greater') # alternative='greater' 测试是否显著大于
    
    print("🧪 配对 t 检验 (Paired t-test):")
    print(f"   t-statistic: {t_stat:.4f}")
    print(f"   p-value:     {p_val_t:.5f}")
    
    if p_val_t < 0.05:
        print("   ✅ 结果显著 (p < 0.05)")
    else:
        print("   ⚠️ 结果不显著 (p >= 0.05)")
        
    print("-" * 50)

    # 2. Wilcoxon 符号秩检验 (Wilcoxon Signed-Rank Test)
    # 非参数检验，不假设正态分布，更适合小样本或有离群值的情况
    w_stat, p_val_w = stats.wilcoxon(ms_se_ib_accs, noib_accs, alternative='greater')
    
    print("🧪 Wilcoxon 符号秩检验 (Wilcoxon Signed-Rank Test):")
    print(f"   statistic: {w_stat:.4f}")
    print(f"   p-value:   {p_val_w:.5f}")
    
    if p_val_w < 0.05:
        print("   ✅ 结果显著 (p < 0.05)")
    else:
        print("   ⚠️ 结果不显著 (p >= 0.05)")

    # 保存结果到 txt
    with open(os.path.join(current_dir, "significance_test_result.txt"), "w", encoding="utf-8") as f:
        f.write(f"Paired t-test p-value: {p_val_t:.5f}\n")
        f.write(f"Wilcoxon p-value:      {p_val_w:.5f}\n")

if __name__ == "__main__":
    main()