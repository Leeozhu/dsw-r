# =============================================================================
# 07b_paired_tests_splitlevel.R — 配对检验口径修正与 CI 补强（增量脚本）
#
# 定位：不修改 07_evaluation.R；读取既有 outputs/metrics/preds/*.rds 重算，
#       修复两处已知口径问题并补强论文需要的区间估计：
#   (1) GLM/GAM 等确定性模型不再按 split×seed 重复计数（伪重复）：
#       凡涉及确定性模型的对比，改为划分级配对（n=5）——随机模型取该划分
#       3 个种子的 PD 均值，与确定性模型在同一划分上配对；
#   (2) E2 细胞间对比按 3 划分×2 种子 = 6 个配对观测报告，并给出配对差的
#       bootstrap 95% CI（论文§五"第二点"改写所需）；
#   (3) 置换检验改为【精确】符号翻转枚举（n≤15 时 2^n 可穷举，替代 07 的
#       Monte Carlo 近似）；
#   (4) 补报符号检验 p 值（回应"效应量大 vs 未达显著"的并置问题）。
#
# 产出（不覆盖 07 的任何文件）：
#   outputs/metrics/paired_tests_splitlevel.csv  —— 全部修正口径的配对检验
#   outputs/metrics/e2_ad_ci.json                —— A/D 对比的论文回填包
#
# 用法：Rscript 07b_paired_tests_splitlevel.R
# 前提：06 已运行（preds rds 齐备）；仅依赖 00_config.R。
# 布局：outputs 根目录自动定位（平铺/平行/嵌套均可），无需手工对位。
# =============================================================================

source("00_config.R")

# ---- outputs 根目录自动定位 ----------------------------------------------------
# 依次尝试三种布局：
#   (1) 平铺：outputs 与脚本同目录（云端形态）；
#   (2) 平行：outputs 为脚本目录的平级（如 dsw-r/{Rscript, outputs}，向上回溯）；
#   (3) 嵌套：outputs 在工作目录的下层某子目录中（向下递归查找目录名）。
# 找不到则停止并打印工作目录，避免"全部 n=0"的静默失败。
locate_outputs_root <- function() {
  sub <- file.path("outputs", "metrics", "preds")
  if (dir.exists(sub)) return(normalizePath("."))
  d <- normalizePath(".")
  for (i in 1:5) {                                  # 向上最多回溯 5 级
    d <- dirname(d)
    if (dir.exists(file.path(d, sub))) return(d)
  }
  ds <- list.dirs(".", recursive = TRUE)            # 向下递归找 preds 目录
  hit <- ds[grepl("outputs/metrics/preds$", ds)]
  if (length(hit)) return(normalizePath(file.path(hit[1], "..", "..", "..")))
  NULL
}

outputs_root <- locate_outputs_root()
if (is.null(outputs_root)) {
  stop("未定位 outputs/metrics/preds。当前工作目录：", getwd(),
       "\n请确认 preds rds 已下载且位于 outputs/metrics/preds，",
       "并把工作目录设在项目目录（或其任意子目录）内。")
}
CFG$paths <- lapply(CFG$paths, function(p) file.path(outputs_root, p))
log_msg("outputs 根目录：%s", outputs_root)
n_rds <- length(list.files(CFG$paths$preds, pattern = "\\.rds$"))
log_msg("preds 目录内 rds 文件数：%d", n_rds)

DETERMINISTIC <- c("GLM", "GAM")          # 每划分仅 seed0 一个预测
N_BOOT <- 10000L
set.seed(42L)

# ---- 载入预测（确定性模型固定取 seed0） --------------------------------------
load_pred <- function(dataset, model, s, k) {
  kk <- if (model %in% DETERMINISTIC) 0L else k
  f <- file.path(CFG$paths$preds, sprintf("%s_%s_split%d_seed%d.rds", dataset, model, s, kk))
  if (!file.exists(f)) return(NULL)
  readRDS(f)
}

# ---- 单个预测对象的测试集 Poisson deviance ------------------------------------
pd_of <- function(p) poisson_deviance(p$y, pmax(p$pred, 1e-10) * p$exposure)

# ---- 配对差向量：三种口径 ------------------------------------------------------
# mode "split_x_seed"：双方均为随机模型，n = n_splits × n_seeds（15）
# mode "e2"         ：E2 细胞，n = e2_splits × e2_seeds（6）
# mode "split_level"：至少一方为确定性模型，n = n_splits（5），
#                    随机方取该划分各种子 PD 的均值（与论文 v2 口径一致）
pair_diffs <- function(dataset, model_a, model_b, cfg = CFG) {
  det <- c(model_a, model_b) %in% DETERMINISTIC
  is_e2 <- grepl("^E2[A-D]_", model_a) || grepl("^E2[A-D]_", model_b)
  if (is_e2) {
    ss <- seq_len(cfg$e2_splits); ks <- seq_len(cfg$e2_seeds)
    d <- c()
    for (s in ss) for (k in ks) {
      pa <- load_pred(dataset, model_a, s, k); pb <- load_pred(dataset, model_b, s, k)
      if (is.null(pa) || is.null(pb)) next
      d <- c(d, pd_of(pa) - pd_of(pb))
    }
    return(list(diffs = d, mode = "e2"))
  }
  if (any(det)) {
    d <- c()
    for (s in seq_len(cfg$n_splits)) {
      pd_one <- function(m) {
        if (m %in% DETERMINISTIC) {
          p <- load_pred(dataset, m, s, 0L); if (is.null(p)) return(NA_real_)
          pd_of(p)
        } else {
          v <- vapply(seq_len(cfg$n_seeds), function(k) {
            p <- load_pred(dataset, m, s, k); if (is.null(p)) NA_real_ else pd_of(p)
          }, numeric(1))
          if (all(is.na(v))) NA_real_ else mean(v, na.rm = TRUE)
        }
      }
      da <- pd_one(model_a); db <- pd_one(model_b)
      if (is.na(da) || is.na(db)) next
      d <- c(d, da - db)
    }
    return(list(diffs = d, mode = "split_level"))
  }
  d <- c()
  for (s in seq_len(cfg$n_splits)) for (k in seq_len(cfg$n_seeds)) {
    pa <- load_pred(dataset, model_a, s, k); pb <- load_pred(dataset, model_b, s, k)
    if (is.null(pa) || is.null(pb)) next
    d <- c(d, pd_of(pa) - pd_of(pb))
  }
  list(diffs = d, mode = "split_x_seed")
}

# ---- 精确符号翻转置换检验（2^n 穷举，n≤15 时无 Monte Carlo 误差） --------------
exact_signflip_p <- function(d) {
  n <- length(d)
  obs <- abs(mean(d))
  S <- 2^n
  stat <- vapply(seq_len(S) - 1, function(i) {
    signs <- ifelse(as.logical(intToBits(i)[seq_len(n)]), 1, -1)
    abs(mean(d * signs))
  }, numeric(1))
  mean(stat >= obs - 1e-12)               # 双尾：|T_perm| >= |T_obs|
}

# ---- 配对差 bootstrap 百分位 CI（对配对重抽样，保持配对结构） -------------------
boot_ci_mean <- function(d, B = N_BOOT) {
  n <- length(d)
  est <- replicate(B, mean(d[sample.int(n, n, replace = TRUE)]))
  unname(quantile(est, c(0.025, 0.975)))
}

# ---- 汇总一个对比 --------------------------------------------------------------
test_one <- function(dataset, model_a, model_b, cfg = CFG) {
  r <- pair_diffs(dataset, model_a, model_b, cfg)
  d <- r$diffs[!is.na(r$diffs)]
  n <- length(d)
  if (n < 3) {
    log_msg("配对不足，跳过：%s vs %s（n=%d）", model_a, model_b, n)
    return(NULL)
  }
  k_pos <- sum(d > 0)
  sign_p <- binom.test(min(k_pos, n - k_pos), n, p = 0.5)$p.value   # 双侧精确符号检验
  wil_p  <- suppressWarnings(wilcox.test(d, mu = 0, exact = (n <= 25))$p.value)
  ci     <- boot_ci_mean(d)
  data.frame(
    dataset = dataset, model_a = model_a, model_b = model_b,
    pairing = r$mode, n_pairs = n,
    mean_diff = mean(d), sd_diff = sd(d),
    frac_pos = k_pos / n, sign_p = sign_p,
    wilcox_p = wil_p, perm_p_exact = exact_signflip_p(d),
    ci_lo = ci[1], ci_hi = ci[2],
    note = if (r$mode == "split_level")
      "n=5 划分级配对：双侧 Wilcoxon 最小可达 p=0.0625，不作传统显著性宣称" else "",
    stringsAsFactors = FALSE
  )
}

# ---- 对比清单：与 07 一致，但剔除 Porto（E4 不入本文） ---------------------------
run_pairwise_tests_v2 <- function(cfg = CFG) {
  pairs <- rbind(
    # RQ1：光谱
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
    # RQ2：E2 细胞对比（6 对）
    c("fremtpl2", "E2A_FTTransformer_adamw", "E2D_SeqOneTrans_adam"),
    c("fremtpl2", "E2B_FTTransformer_adam",  "E2D_SeqOneTrans_adam"),
    c("fremtpl2", "E2C_SeqOneTrans_adamw",   "E2D_SeqOneTrans_adam")
  )
  res <- do.call(rbind, lapply(seq_len(nrow(pairs)), function(i)
    test_one(pairs[i, 1], pairs[i, 2], pairs[i, 3], cfg)))
  write.csv(res, file.path(cfg$paths$metrics, "paired_tests_splitlevel.csv"),
            row.names = FALSE)
  log_msg("修正口径配对检验已写盘：paired_tests_splitlevel.csv（%d 行）", nrow(res))
  res
}

# ---- A/D 论文回填包 -------------------------------------------------------------
write_ad_package <- function(res, cfg = CFG) {
  if (is.null(res) || !nrow(res)) { log_msg("配对检验结果为空，跳过 A/D 回填包"); return(invisible(NULL)) }
  row <- res[res$model_a == "E2A_FTTransformer_adamw" &
             res$model_b == "E2D_SeqOneTrans_adam", ]
  if (!nrow(row)) { log_msg("未找到 A/D 行"); return(invisible(NULL)) }
  pkg <- list(
    comparison = "E2 cell A (healthy) minus cell D (Q/K selective death)",
    n_pairs = row$n_pairs,
    mean_diff_pd = row$mean_diff,
    sd_diff = row$sd_diff,
    ci95 = c(lo = row$ci_lo, hi = row$ci_hi),
    wilcox_p = row$wilcox_p, signflip_p_exact = row$perm_p_exact,
    sign_p = row$sign_p,
    direction_note = "正差 = A 劣于 D；当前点估计为正（A 的 PD 更高），即失效细胞 D 的指标甚至略优于健康基线",
    precommitted_readings = list(
      ci_wide_includes_0 = paste(
        "写法一（CI 宽、含 0）：强调'无可检出差异'——在 n=6 下差异上界仍仅为",
        "CI 上限量级，且点估计方向为 D 略优；审查程序对 D 无警报输出。"),
      ci_narrow_excludes_0_D_better = paste(
        "写法二（CI 窄、不含 0 且为正）：结论增强——失效模型的指标系统性地更好，",
        "是审查失灵的极端形态。"),
      ci_narrow_excludes_0_D_worse = paste(
        "写法三（CI 窄、不含 0 且为负）：放弃'无可检出差异'表述，退守'差异量级",
        "（|mean_diff|）远小于失效的结构严重性，且远低于任何警报阈值'——",
        "此情形需在正文中显式披露并重新措辞。")
    )
  )
  save_json(pkg, file.path(cfg$paths$metrics, "e2_ad_ci.json"))
  log_msg("A/D 回填包已写盘：e2_ad_ci.json | mean=%.5f CI=[%.5f, %.5f] perm_p=%.4f",
          row$mean_diff, row$ci_lo, row$ci_hi, row$perm_p_exact)
  invisible(pkg)
}

if (sys.nframe() == 0L) {
  init_dirs()
  res <- run_pairwise_tests_v2()
  write_ad_package(res)
  print(res[, c("model_a", "model_b", "pairing", "n_pairs", "mean_diff",
                "frac_pos", "sign_p", "wilcox_p", "perm_p_exact", "ci_lo", "ci_hi")],
        row.names = FALSE, digits = 5)
}
