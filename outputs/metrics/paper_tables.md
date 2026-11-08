# 论文结果表（自动生成）

生成时间：2026-11-03 08:58:06.767711

## 表 1  透明度-性能光谱（freMTPL2，Poisson deviance / Gini / 校准）

| 层级 | 模型 | n | PD | Gini（频率口径） | 校准 RMSE |
|---|---|---|---|---|---|
| L0 | GLM | 5 | 0.2408 ± 0.0006 | 0.2981 ± 0.0047 | 0.063 ± 0.007 |
| L1 | GAM | 5 | 0.2405 ± 0.0006 | 0.2996 ± 0.0051 | 0.068 ± 0.018 |
| L2 | XGBoost | 15 | 0.2362 ± 0.0005 | 0.3320 ± 0.0072 | 0.068 ± 0.011 |
| L3 | MLP_S | 15 | 0.2401 ± 0.0006 | 0.3061 ± 0.0052 | 0.075 ± 0.021 |
| L4 | FTTransformer | 15 | 0.2401 ± 0.0010 | 0.3072 ± 0.0068 | 0.134 ± 0.039 |
| L4 | MLP_L | 15 | 0.2408 ± 0.0005 | 0.2992 ± 0.0072 | 0.129 ± 0.043 |
| L4 | TCTParallel | 15 | 0.2394 ± 0.0006 | 0.3094 ± 0.0052 | 0.123 ± 0.051 |
| — | TCT_abl_avg_fusion | 15 | 0.2395 ± 0.0005 | 0.3083 ± 0.0066 | 0.126 ± 0.048 |
| — | TCT_abl_no_attn | 15 | 0.2398 ± 0.0003 | 0.3091 ± 0.0055 | 0.100 ± 0.043 |
| — | TCT_abl_no_tcn | 15 | 0.2399 ± 0.0006 | 0.3096 ± 0.0074 | 0.132 ± 0.036 |

## 表 2  配对检验（跨划分 x 种子的配对差；正差 = A 劣于 B）

| 数据集 | A | B | 配对数 | 平均差 | Wilcoxon p | 置换 p |
|---|---|---|---|---|---|---|
| fremtpl2 | GAM | GLM | 15 | -0.00035 | 6.104e-05 | 0 |
| fremtpl2 | XGBoost | GAM | 15 | -0.00424 | 6.104e-05 | 0 |
| fremtpl2 | MLP_S | XGBoost | 15 | 0.00384 | 6.104e-05 | 0 |
| fremtpl2 | MLP_L | MLP_S | 15 | 0.00072 | 0.0004272 | 0.0005 |
| fremtpl2 | FTTransformer | MLP_L | 15 | -0.00068 | 0.002625 | 0.0011 |
| fremtpl2 | TCTParallel | MLP_S | 15 | -0.00072 | 0.00116 | 0.0005 |
| fremtpl2 | TCTParallel | GLM | 15 | -0.00147 | 6.104e-05 | 0 |
| fremtpl2 | TCT_abl_no_attn | TCTParallel | 15 | 0.00040 | 0.02155 | 0.0283 |
| fremtpl2 | TCT_abl_no_tcn | TCTParallel | 15 | 0.00059 | 0.002625 | 0.0022 |
| fremtpl2 | TCT_abl_avg_fusion | TCTParallel | 15 | 0.00015 | 0.3894 | 0.3386 |
| fremtpl2 | E2A_FTTransformer_adamw | E2D_SeqOneTrans_adam | 6 | 0.00028 | 0.6875 | 0.5285 |
| fremtpl2 | E2B_FTTransformer_adam | E2D_SeqOneTrans_adam | 6 | -0.00045 | 0.03125 | 0.0295 |
| fremtpl2 | E2C_SeqOneTrans_adamw | E2D_SeqOneTrans_adam | 6 | 0.00051 | 0.1562 | 0.1231 |
| porto | MLP_S | MLP_S_DOUBLECOMP | 0 | NA | NA | NA |
| porto | MLP_S | MLP_S_CALIBRATED | 0 | NA | NA | NA |

## 表 3  受控失效注入：指标与内部健康的分离（RQ2 核心证据）

| 细胞 | 输入 | 优化器 | PD（均值） | Gini（均值） | 参数死亡 | Q/K 选择性死亡 | 注意力熵坍塌 |
|---|---|---|---|---|---|---|---|
| A | tokenizer 正确 | adamw | 0.2406 | 0.3062 | 否 | 否 | 否 |
| B | tokenizer 正确 | adam | 0.2399 | 0.3109 | 否 | 否 | 否 |
| C | seq_len=1 误用 | adamw | 0.2408 | 0.3015 | 否 | 否 | NA |
| D | seq_len=1 误用 | adam | 0.2403 | 0.3061 | 否 | 是 | NA |

## 表 4  分层审计清单（RQ3 交付物模板）

| 层级 | 必查项 | 监管依据 | 检出失效类型 | 本研究证据 |
|---|---|---|---|---|
| 全部 | 测试集指标 + 频率口径 Gini + 校准曲线 | Solvency II Art.242；EIOPA AI 意见 | 全局性能/校准 | 表 1 |
| L2+ | 置换重要性 + 精算先验方向一致性 | NAIC 示范公告 | 特征误用 | 08 输出 |
| L3+ | 训练管线配置审计（采样/加权/损失） | EIOPA 意见：文档化 | Porto 型管线错误 | E4 对比 |
| L4 | 组件级审计：Q/K/V/O 范数、分支激活、门控、注意力熵 | 相称性原则 | 静默坍塌、结构恒等化 | 表 3 |
| L4 | 消融贡献审计 | 独立验证原则 | 死分支/冗余复杂度 | TCT_abl_* 配对 |

