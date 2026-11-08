# =============================================================================
# 10_report.R — 自动生成论文表格（全部从 outputs/ 读取）
# 表 3 增加"注意力熵坍塌"列
# =============================================================================

source("00_config.R")

fmt_pm <- function(m, s, digits = 4)
  ifelse(is.na(s), sprintf("%.*f", digits, m),
         sprintf("%.*f ± %.*f", digits, m, digits, s))

build_paper_tables <- function(cfg = CFG) {
  summ_f <- file.path(cfg$paths$metrics, "metrics_summary.csv")
  if (!file.exists(summ_f)) stop("先运行 07_evaluation.R")
  summ <- read.csv(summ_f)
  tests_f <- file.path(cfg$paths$metrics, "paired_tests.csv")
  tests <- if (file.exists(tests_f)) read.csv(tests_f) else NULL
  e2_f <- file.path(cfg$paths$metrics, "e2_detection_matrix.csv")
  e2 <- if (file.exists(e2_f)) read.csv(e2_f) else NULL

  L <- c("# 论文结果表（自动生成）", "", sprintf("生成时间：%s", Sys.time()), "")

  # ---- 表 1：透明度-性能光谱 --------------------------------------------------
  L <- c(L, "## 表 1  透明度-性能光谱（freMTPL2，Poisson deviance / Gini / 校准）", "")
  e1 <- summ[summ$dataset == "fremtpl2" & !grepl("^E2", summ$model), ]
  e1$level <- factor(MODEL_SPECTRUM$level[match(e1$model, MODEL_SPECTRUM$model)],
                     levels = c("L0", "L1", "L2", "L3", "L4"))
  e1 <- e1[order(e1$level), ]
  L <- c(L, "| 层级 | 模型 | n | PD | Gini（频率口径） | 校准 RMSE |",
         "|---|---|---|---|---|---|")
  for (i in seq_len(nrow(e1)))
    L <- c(L, sprintf("| %s | %s | %d | %s | %s | %s |",
                      ifelse(is.na(e1$level[i]), "—", as.character(e1$level[i])),
                      e1$model[i], e1$n_runs[i],
                      fmt_pm(e1$poisson_deviance_mean[i], e1$poisson_deviance_sd[i]),
                      fmt_pm(e1$gini_freq_mean[i], e1$gini_freq_sd[i]),
                      fmt_pm(e1$calib_rmse_mean[i], e1$calib_rmse_sd[i], 3)))
  L <- c(L, "")

  # ---- 表 2：配对检验 --------------------------------------------------------
  if (!is.null(tests)) {
    L <- c(L, "## 表 2  配对检验（跨划分 x 种子的配对差；正差 = A 劣于 B）", "",
           "| 数据集 | A | B | 配对数 | 平均差 | Wilcoxon p | 置换 p |",
           "|---|---|---|---|---|---|---|")
    for (i in seq_len(nrow(tests)))
      L <- c(L, sprintf("| %s | %s | %s | %d | %.5f | %.4g | %.4g |",
                        tests$dataset[i], tests$model_a[i], tests$model_b[i],
                        tests$n_pairs[i], tests$mean_diff[i],
                        tests$wilcox_p[i], tests$perm_p[i]))
    L <- c(L, "")
  }

  # ---- 表 3：E2 失效注入 2x2 -------------------------------------------------
  if (!is.null(e2)) {
    L <- c(L, "## 表 3  受控失效注入：指标与内部健康的分离（RQ2 核心证据）", "",
           "| 细胞 | 输入 | 优化器 | PD（均值） | Gini（均值） | 参数死亡 | Q/K 选择性死亡 | 注意力熵坍塌 |",
           "|---|---|---|---|---|---|---|---|")
    for (cid in c("A", "B", "C", "D")) {
      g <- e2[e2$cell == cid, ]
      if (!nrow(g)) next
      mm <- summ[summ$model == sprintf("E2%s_%s_%s", cid, g$model[1], g$optimizer[1]), ]
      L <- c(L, sprintf("| %s | %s | %s | %s | %s | %s | %s | %s |",
                        cid,
                        ifelse(grepl("SeqOne", g$model[1]), "seq_len=1 误用", "tokenizer 正确"),
                        g$optimizer[1],
                        if (nrow(mm)) sprintf("%.4f", mm$poisson_deviance_mean[1]) else "—",
                        if (nrow(mm)) sprintf("%.4f", mm$gini_freq_mean[1]) else "—",
                        ifelse(any(g$params_dead == TRUE), "是", "否"),
                        ifelse(any(g$qk_selective_death == "TRUE"), "是", "否"),
                        ifelse(any(g$attention_entropy_collapse == "TRUE"), "是", "否")))
    }
    L <- c(L, "")
  }

  # ---- 表 4：审计清单-条文映射（RQ3 模板） -----------------------------------
  L <- c(L, "## 表 4  分层审计清单（RQ3 交付物模板）", "",
         "| 层级 | 必查项 | 监管依据 | 检出失效类型 | 本研究证据 |",
         "|---|---|---|---|---|",
         "| 全部 | 测试集指标 + 频率口径 Gini + 校准曲线 | Solvency II Art.242；EIOPA AI 意见 | 全局性能/校准 | 表 1 |",
         "| L2+ | 置换重要性 + 精算先验方向一致性 | NAIC 示范公告 | 特征误用 | 08 输出 |",
         "| L3+ | 训练管线配置审计（采样/加权/损失） | EIOPA 意见：文档化 | Porto 型管线错误 | E4 对比 |",
         "| L4 | 组件级审计：Q/K/V/O 范数、分支激活、门控、注意力熵 | 相称性原则 | 静默坍塌、结构恒等化 | 表 3 |",
         "| L4 | 消融贡献审计 | 独立验证原则 | 死分支/冗余复杂度 | TCT_abl_* 配对 |",
         "")

  out <- file.path(cfg$paths$metrics, "paper_tables.md")
  writeLines(L, out)
  log_msg("论文表格已生成：%s", out)
  invisible(out)
}

if (sys.nframe() == 0L) { init_dirs(); build_paper_tables() }
