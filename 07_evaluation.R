# =============================================================================
# 07_evaluation.R — 汇总指标 + 跨划分 x 种子配对检验
# 修复：列对齐（bind_rows 兼容 poisson/binary）；阈值来源（rds > 逐划分 json > 0.5）
# =============================================================================

source("00_config.R")

# ---- 全局阈值：Porto 逐划分阈值文件的回退 --------------------------------------
get_global_threshold <- function(dataset, split, cfg = CFG) {
  f <- file.path(cfg$paths$metrics, sprintf("porto_correction_split%d.json", split))
  if (dataset == "porto" && file.exists(f))
    return(jsonlite::read_json(f)$threshold)
  0.5
}

# ---- 汇总所有预测 -> 长表 -----------------------------------------------------
collect_metrics <- function(cfg = CFG) {
  files <- list.files(cfg$paths$preds, pattern = "\\.rds$", full.names = TRUE)
  if (!length(files)) return(invisible(NULL))
  rows <- lapply(files, function(f) {
    p <- readRDS(f)
    thr <- if (!is.null(p$threshold)) p$threshold
           else get_global_threshold(p$dataset, p$split, cfg)
    task <- if (!is.null(p$exposure)) "poisson" else "binary"
    m <- eval_task(task, p$y, p$pred, p$exposure, thr)
    data.frame(dataset = p$dataset, model = p$model, split = p$split,
               seed = p$seed, as.list(m), check.names = FALSE)
  })
  # 不同任务指标列不同：bind_rows 自动补 NA（无 dplyr 时手工补列）
  out <- if (requireNamespace("dplyr", quietly = TRUE)) {
    dplyr::bind_rows(rows)
  } else {
    all_cols <- unique(unlist(lapply(rows, names)))
    rows <- lapply(rows, function(r) {
      for (cc in setdiff(all_cols, names(r))) r[[cc]] <- NA
      r[, all_cols, drop = FALSE]
    })
    do.call(rbind, rows)
  }
  write.csv(out, file.path(cfg$paths$metrics, "metrics_long.csv"), row.names = FALSE)
  out
}

summarize_metrics <- function(long = NULL, cfg = CFG) {
  if (is.null(long)) long <- read.csv(file.path(cfg$paths$metrics, "metrics_long.csv"))
  num_cols <- names(long)[sapply(long, is.numeric) & !names(long) %in% c("split", "seed")]
  agg <- do.call(rbind, lapply(split(long, list(long$dataset, long$model), drop = TRUE),
    function(g) {
      row <- data.frame(dataset = g$dataset[1], model = g$model[1], n_runs = nrow(g))
      for (v in num_cols) {
        if (all(is.na(g[[v]]))) next
        row[[paste0(v, "_mean")]] <- mean(g[[v]], na.rm = TRUE)
        row[[paste0(v, "_sd")]]   <- sd(g[[v]], na.rm = TRUE)
      }
      row
    }))
  agg$level <- MODEL_SPECTRUM$level[match(agg$model, MODEL_SPECTRUM$model)]
  write.csv(agg, file.path(cfg$paths$metrics, "metrics_summary.csv"), row.names = FALSE)
  agg
}

# ---- 配对检验 ----------------------------------------------------------------
paired_test <- function(dataset, model_a, model_b, cfg = CFG, n_perm = 10000L) {
  load_pred <- function(m, s, k) {
    kk <- if (m %in% c("GLM", "GAM")) 0L else k
    f <- file.path(cfg$paths$preds, sprintf("%s_%s_split%d_seed%d.rds", dataset, m, s, kk))
    if (!file.exists(f)) return(NULL)
    readRDS(f)
  }
  diffs <- c()
  for (s in seq_len(cfg$n_splits)) for (k in seq_len(cfg$n_seeds)) {
    pa <- load_pred(model_a, s, k); pb <- load_pred(model_b, s, k)
    if (is.null(pa) || is.null(pb)) next
    if (!is.null(pa$exposure)) {
      dev_one <- function(y, mu) { mu <- pmax(mu, 1e-10)
        2 * (ifelse(y == 0, mu, y * log(y / mu) - (y - mu))) }
      d <- dev_one(pa$y, pa$pred * pa$exposure) - dev_one(pb$y, pb$pred * pb$exposure)
    } else {
      d <- (pa$y - pa$pred)^2 - (pb$y - pb$pred)^2
    }
    diffs <- c(diffs, mean(d))
  }
  diffs <- diffs[!is.na(diffs)]
  if (length(diffs) < 3)
    return(data.frame(dataset, model_a, model_b, n_pairs = length(diffs),
                      mean_diff = NA, wilcox_p = NA, perm_p = NA))
  wil <- suppressWarnings(wilcox.test(diffs, mu = 0)$p.value)
  set.seed(1L)
  perm_stats <- replicate(n_perm, mean(diffs * sample(c(-1, 1), length(diffs), TRUE)))
  perm <- mean(abs(perm_stats) >= abs(mean(diffs)))
  data.frame(dataset, model_a, model_b, n_pairs = length(diffs),
             mean_diff = mean(diffs), wilcox_p = wil, perm_p = perm)
}

run_pairwise_tests <- function(cfg = CFG) {
  pairs <- rbind(
    # RQ1：光谱相邻对比
    c("fremtpl2", "GAM",  "GLM"),
    c("fremtpl2", "XGBoost", "GAM"),
    c("fremtpl2", "MLP_S", "XGBoost"),
    c("fremtpl2", "MLP_L", "MLP_S"),
    c("fremtpl2", "FTTransformer", "MLP_L"),
    c("fremtpl2", "TCTParallel", "MLP_S"),
    c("fremtpl2", "TCTParallel", "GLM"),
    # RQ1 消融
    c("fremtpl2", "TCT_abl_no_attn", "TCTParallel"),
    c("fremtpl2", "TCT_abl_no_tcn",  "TCTParallel"),
    c("fremtpl2", "TCT_abl_avg_fusion", "TCTParallel"),
    # RQ2：E2 细胞对比
    c("fremtpl2", "E2A_FTTransformer_adamw", "E2D_SeqOneTrans_adam"),
    c("fremtpl2", "E2B_FTTransformer_adam",  "E2D_SeqOneTrans_adam"),
    c("fremtpl2", "E2C_SeqOneTrans_adamw",   "E2D_SeqOneTrans_adam"),
    # E4：失效管线 vs 纠正管线
    c("porto", "MLP_S", "MLP_S_DOUBLECOMP"),
    # E4 校准前后
    c("porto", "MLP_S", "MLP_S_CALIBRATED")
  )
  res <- do.call(rbind, lapply(seq_len(nrow(pairs)), function(i)
    paired_test(pairs[i, 1], pairs[i, 2], pairs[i, 3], cfg)))
  write.csv(res, file.path(cfg$paths$metrics, "paired_tests.csv"), row.names = FALSE)
  res
}

if (sys.nframe() == 0L) {
  init_dirs(); collect_metrics(); summarize_metrics(); run_pairwise_tests()
}
