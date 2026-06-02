import os
import json
import pandas as pd
import numpy as np

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, "results_ma", "noib_all")
    csv_path = os.path.join(current_dir, "results_ma", "summary_noib_ma.csv")
    
    print(f"📂 Scanning JSON files in: {results_dir}")
    
    fixed_results = []
    
    for filename in os.listdir(results_dir):
        if filename.endswith("_noib_ma.json"):
            filepath = os.path.join(results_dir, filename)
            with open(filepath, "r") as f:
                data = json.load(f)
                
            subject_id = data["subject"]
            fold_results = data["fold_results"]
            
            # fold_results 是一个 list of lists
            # 每个 list 包含 [total_loss, class_loss, eeg_loss, class_acc, eeg_acc]
            # 我们需要的是 class_acc，通常是第 4 个元素 (index 3)
            # 但为了保险，我们取倒数第二个 (假设 metrics 顺序固定)
            # 或者更稳妥：通常 acc 是 < 1 的，loss 可能 > 1
            
            # 让我们假设标准顺序: [loss, class_loss, eeg_loss, class_acc, eeg_acc]
            # Index 3 是 class_acc
            
            accs = []
            for res in fold_results:
                # 尝试找到准确率
                # 如果有 5 个元素，取 index 3
                if len(res) >= 4:
                    acc = res[3] 
                else:
                    # 如果只有 2 个元素 [loss, acc]，取 index 1
                    acc = res[1]
                accs.append(acc)
            
            mean_acc = np.mean(accs)
            
            print(f"   Subject {subject_id:02d}: Raw={fold_results[0]} -> Accs={accs} -> Mean={mean_acc:.4f}")
            
            fixed_results.append({
                "subject": subject_id,
                "noib_ma_acc": mean_acc
            })
            
    # 排序并保存
    fixed_results.sort(key=lambda x: x["subject"])
    df = pd.DataFrame(fixed_results)
    
    print("\n" + "=" * 40)
    print("📊 Fixed STA-Net (NoIB) MA Results")
    print("=" * 40)
    print(df.to_string(index=False))
    print("-" * 40)
    print(f"Average: {df['noib_ma_acc'].mean():.4f}")
    
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Fixed CSV saved to: {csv_path}")

if __name__ == "__main__":
    main()