import subprocess
import sys
import time

# ==================== 1. 配置你想跑的参数网格 ====================
# 这里列出你想尝试的所有组合
experiments = [
    # 组 1: 微调 Beta (在 1e-4 附近)
    {"beta": 5e-5, "latent_dim": 128},
    {"beta": 2e-4, "latent_dim": 128},

    # 组 2: 尝试不同的隐变量维度 (压缩得更狠 vs 给更多空间)
    {"beta": 1e-4, "latent_dim": 64},  # 维度减半
    {"beta": 1e-4, "latent_dim": 256},  # 维度加倍

    # 组 3: 换个随机种子试试运气 (验证 1e-4 的稳定性)
    {"beta": 1e-4, "latent_dim": 128, "seed": 123},
    {"beta": 1e-4, "latent_dim": 128, "seed": 2024},
]

# 你的训练脚本名字
script_name = "run_sta_net_subject01_MI_ib.py"
# ===============================================================

total = len(experiments)
print(f"🚀 准备开始自动化训练，共有 {total} 组实验等待运行...\n")

for i, params in enumerate(experiments):
    print(f"[{i + 1}/{total}] 正在运行: {params} ...")

    # 构建命令
    # 使用 sys.executable 确保用的是当前环境的 python (tf-gpu)
    cmd = [sys.executable, script_name]

    # 自动把参数加进去
    for key, value in params.items():
        cmd.append(f"--{key.replace('_', '-')}")  # 把 latent_dim 变成 --latent-dim
        cmd.append(str(value))

    start_time = time.time()

    try:
        # 调用子进程运行训练脚本
        # check=True 会在训练报错时抛出异常，防止错误的实验继续刷屏
        subprocess.run(cmd, check=True)

        duration = time.time() - start_time
        print(f"✅ 第 {i + 1} 组实验完成，耗时: {duration / 60:.2f} 分钟。\n")
        print("-" * 50)

    except subprocess.CalledProcessError:
        print(f"❌ 第 {i + 1} 组实验报错！跳过该组，继续下一组...\n")
        print("-" * 50)
    except KeyboardInterrupt:
        print("\n🛑 用户手动停止任务。")
        break

print("\n🎉 所有计划任务已执行完毕！请查看 experiment_results.csv 获取汇总结果。")