# dsw-r v2.3：透明度光谱与模型可验证性——R 实验管线

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24076358.svg)](https://doi.org/10.5281/zenodo.24076358)

## 引用本成果（论文参考文献）

> Zhu, H. (2026). *dsw-r: Transparency Spectrum and Model Verifiability — R Experiment Pipeline* (Version 2.3) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.24076358

- **Zenodo 归档（含全部代码与实验结果）**：https://doi.org/10.5281/zenodo.24076358
- **Gitee 镜像**：https://gitee.com/leeozhu/dsw-r
- **GitHub**：https://github.com/Leeozhu/dsw-r

```bibtex
@software{zhu2026dswr,
  author    = {Zhu, Hang},
  title     = {dsw-r: Transparency Spectrum and Model Verifiability — R Experiment Pipeline},
  year      = {2026},
  publisher = {Zenodo},
  version   = {2.3},
  doi       = {10.5281/zenodo.24076358},
  url       = {https://doi.org/10.5281/zenodo.24076358}
}
```

> **版本标识（2026-10 固化）**
> - **代码版本**：v2.3 = v2 叙事重构管线 + 两轮匿名审校修复 + 冒烟实测修复
>   - v2.1：修复 GLM offset 丢失、Density=0、张量切片赋值、`nn_sequential(!!!)` 等
>   - v2.2：修复 `run_all` 的 `init_dirs()` 先于 `source(00_config)`（P0.1）、
>     `collect_metrics` 列错位（P0.2）、恢复 TCT 严格消融、E4 DOUBLECOMP 跑全划分、
>     pos_weight/isotonic 逐划分独立拟合、新增注意力熵审计（检出"结构恒等化"）
>   - v2.3：修复 `for (b in self$blocks)` 遍历 nn_module_list（R6 对象不可迭代，
>     报错 `invalid for() loop sequence`），改为 `self$blocks[[i]]` 索引访问（共 5 处）；
>     **目录扁平化**：所有脚本移至根目录、source() 路径去掉 `R/` 前缀
> - **对应方案**：《研究方案：偿付能力监管下 AI 定价模型的可验证性》**v1.2**
> - **数据集**：freMTPL2（主战场）+ Porto（E4 案例）；**已弃用 Allstate**
> - **如何确认你手上是这份**：脚本**直接位于 dsw-r/ 根目录**（无 R/ 子目录），
>   有 `06_run_experiments.R`、`11_preregister.R`，没有 `06_ablation.R` /
>   `02b_prep_allstate.R`；`09_audit.R` 含 `compute_attention_entropy`；
>   `05_models_torch.R` 中检索不到 `for (b in self` —— 满足即为本版。

叙事核心：**监管可验证性义务 vs 深度模型不透明性**；实验按透明度光谱（L0–L4）组织，
失效通过 2×2 受控注入复现，审计工具的检测能力本身就是研究对象。

## 目录

```
dsw-r/
├── 00_config.R            # 配置、指标（频率 Gini/PD/校准 RMSE）、光谱注册表
├── 01_prep_freMTPL2.R     # 主战场数据：去 raw+log 冗余、factor 保留、5 份划分
├── 02_prep_porto.R        # E4 案例数据：-1 缺失保留信号 + was_missing 指示列
├── 03_glm_baseline.R      # L0：正确规格 GLM（ns 样条 + factor + offset）
├── 04_gbm_models.R        # L2：XGBoost（poisson+base_margin / logistic）
├── 04a_gam_models.R       # L1：GAM/GA2M（mgcv，含受控交互 + 效应曲线写盘）
├── 05_models_torch.R      # L3/L4：MLP_S/MLP_L/FT/SeqOne/TCN/TCT + 双优化器训练器
├── 06_run_experiments.R   # 实验编排：E1 光谱、E2 失效注入（2x2）、E4 Porto
├── 07_evaluation.R        # 汇总 + 配对检验（Wilcoxon + 符号翻转置换）
├── 08_interpret.R         # 置换重要性 + 精算先验方向一致性（如实口径）
├── 09_audit.R             # 组件级审计 + 失效签名检测 + E2 检测矩阵
├── 10_report.R            # 论文表格自动生成（光谱表/检验表/2x2 表/清单模板）
├── 99_smoke_test.R        # 合成数据端到端冒烟测试（先跑这个！）
├── data/raw/              # 原始 CSV（freMTPL2 也可用 CASdatasets 自动加载）
├── outputs/               # 全部结果自动写盘（metrics / weights / figures）
└── run_all.R              # 一键流程
```

## 环境安装（R ≥ 4.3）

```r
install.packages(c("torch", "xgboost", "mgcv", "jsonlite"))
install.packages("CASdatasets")   # freMTPL2 数据源（或放 CSV 到 data/raw/）
torch::install_torch()            # LibTorch（CPU/MPS）
```

## 跑通顺序

```bash
Rscript 11_preregister.R    # 0. E2 判定标准预注册（必须在任何 E2 实验之前，§6.7）
Rscript 99_smoke_test.R     # 1. 冒烟测试（几分钟，验证 torch API 兼容）
Rscript run_all.R quick       # 2. 快速模式：freMTPL2，2 划分 x 1 种子
Rscript run_all.R             # 3. freMTPL2 全流程（E1 光谱 + E2 失效注入）
Rscript run_all.R porto       # 4. E4 Porto 案例（需 data/raw/porto_train.csv）
Rscript run_all.R all         # 5. 全部
```

注：`run_all.R` 会在加载数据前自动调用 `preregister_e2()`；预注册文件一旦生成即锁定，
重复运行不会覆盖（研究方案 §6.7 的 HARKing 防线）。

## 实验设计与研究问题的对应

| 实验 | 脚本 | 研究问题 | 关键产出 |
|------|------|----------|----------|
| E1 透明度-性能光谱 | 03/04/04a/06 | RQ1 | 表 1（光谱表）、配对检验 |
| E2 失效注入 2×2 | 06 | RQ2 | 表 3：指标 vs 内部健康分离的证据 |
| E3 审计检测矩阵 | 09 | RQ2 | e2_detection_matrix.csv、失效签名布尔判定 |
| E4 Porto 案例 | 06 | RQ2/RQ3 | 失效管线 vs 纠正管线对比、校准前后指标 |
| RQ3 审计清单 | 10 | RQ3 | 表 4（清单-条文映射模板） |

## E2 的 2×2 因子设计（本研究方法论核心）

| | AdamW（健康） | Adam 耦合 L2（病理） |
|---|---|---|
| **FT tokenizer（正确输入）** | 细胞 A：健康基线 | 细胞 B：预期不坍塌（证伪旧论文"必然收敛"） |
| **seq_len=1（误用）** | 细胞 C：结构失效但数值健康 | 细胞 D：复现静默坍塌（Q/K 压至精度地板） |

若 A 与 D 的测试指标接近而审计签名截然不同，即证明"指标无法区分健康与失效"——
这是整套审计清单存在的理由。

## 与旧版管线的关键差异

- 叙事从"架构对比"改为"透明度光谱 × 失效检测"；TCT 降级为 L4 代表之一
- 新增 GAM 半透明层（L1）与 MLP 大小两档（L3/L4 的复杂度梯度）
- 新增双优化器支持（AdamW 主实验 / 耦合 Adam 失效注入）
- Porto 拆为"失效复现（教学案例）"与"纠正管线（单一补偿 + isotonic 校准）"两支
- Allstate 舍弃（匿名特征与内部语义验证叙事不兼容）
- 审计输出从"权重范数表"升级为"失效签名布尔判定 + 检测矩阵"

## 诚实边界

- 本套脚本经逐行审查与括号级静态检查；冒烟测试已暴露并修复 nn_module_list 遍历问题
  （v2.3）。若后续仍报错，最可能出现在 torch 张量索引细节（`h[, 1, ]`、`torch_narrow`），
  把报错信息发来即可快速修复。
- `nn_multihead_attention` 需要较新版 torch 包（≥ 0.10）；旧版缺 `batch_first` 时告知，我提供兼容写法。
- 解剖旧 Python 检查点（方案第 0 阶段）仍需单独的 Python 脚本，不属于本管线。
