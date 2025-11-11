### STA-Net IB 模型架构图

下面用 Mermaid 展示 `sta_net_ib` 的整体数据流与关键信息瓶颈模块（IB）位置。

```mermaid
flowchart LR
  subgraph Inputs
    EEG[EEG Input\n(16,16,600,1)]
    FNIRS[fNIRS Input\n(11,16,16,30,2)]
  end

  subgraph Block1[Conv Block 1]
    direction LR
    EEG1[EEG Conv3D + BN + ELU]
    FNIRS1[fNIRS Conv3D + BN + ELU]
    EEFUSION1[EEGFusion Conv3D + BN + ELU]
    FGA1[FGA (channel pooling + TAP + sigmoid)\nGuided fusion]
    EEG --> EEG1
    FNIRS --> FNIRS1
    EEGFUSION1 --> FGA1
    EEG1 --> FGA1
    FNIRS1 --> FGA1
  end

  subgraph Block2[Conv Block 2 + GAP]
    direction LR
    EEG2[EEG Conv3D + BN + ELU -> GAP]
    FNIRS2[fNIRS Conv3D + BN + ELU -> GAP]
    EEFUSION2[EEGFusion Conv3D + BN + ELU -> GAP]
    FGA1 --> EEFUSION2
    EEG1 --> EEG2
    FNIRS1 --> FNIRS2
  end

  subgraph EF_Attn[EEG–fNIRS Attention]
    direction TB
    Q[Flatten + Dense(emb=256)]
    K[pos embedding + Dense(emb=256)]
    MHA[Multi-Head Attention (heads=10, d_model=256, dropout=0.5)]
    PLCC["PLCC loss (1 - pearson_r)"]
  end

  EEFUSION2 -->|q from EEGFusion| Q --> MHA
  FNIRS2 -->|k from fNIRS| K --> MHA
  MHA -->|fusion_out| EF_FUSION[EEGFusion feature (256, ELU, Dense 256 ELU)]
  MHA -->|fnirs_weighted| EF_FNIRS[fNIRS feature (256, ELU, Dense 256 ELU)]
  Q --> PLCC
  MHA --> PLCC

  subgraph IBs[Information Bottleneck (beta, latent_dim)]
    direction LR
    IB_EEG[IB(eeg_ib)]
    IB_EF[IB(eegfusion_ib)]
    IB_FNIRS[IB(fnirs_ib)]
  end

  EEG2 -->|Flatten + Dense 256 ELU| IB_EEG
  EF_FUSION --> IB_EF
  EF_FNIRS --> IB_FNIRS

  subgraph Heads[Predictions + Weighting]
    direction LR
    EEG_SOFT[EEG head: Dense 2 -> softmax\n(name: eeg_output)]
    EF_SOFT[EEGFusion head: Dense 2 -> softmax]
    FNIRS_SOFT[fNIRS head: Dense 2 -> softmax]
    WEIGHT["p_weight = softmax(Dense(1) on IB features)\nthen expand_dims"]
    CONCAT[Concatenate axis=1]
    MUL[Multiply with p_weight]
    SUM[ReduceSum axis=1\n(name: class_output)]
  end

  IB_EEG --> EEG_SOFT
  IB_EF --> EF_SOFT --> CONCAT
  IB_FNIRS --> FNIRS_SOFT --> CONCAT
  IB_EF -->|Dense(1)| WEIGHT
  IB_FNIRS -->|Dense(1)| WEIGHT
  CONCAT --> MUL --> SUM
```

要点：
- 信息瓶颈层位置：`eegfusion_ib`、`fnirs_ib`、`eeg_ib` 分别作用在融合特征、fNIRS 特征、EEG 特征上，KL 损失按 `beta` 加权加入总损失。
- EF 注意力：以 EEGFusion 为查询，fNIRS 为键值，加入 PLCC 相关性损失促进跨模态对齐。
- 最终分类：EEGFusion 与 fNIRS 两个分支的 softmax 结果经权重 `p_weight` 自适应加权后汇总为 `class_output`；EEG 分支单独输出 `eeg_output` 作为辅助任务。

文件对应实现：`sta.py` 中的 `sta_net_ib`、`InformationBottleneck`、`e_f_attention`、`conv_block`、`fga` 等类与函数。


