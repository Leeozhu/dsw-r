# =============================================================================
# 01_prep_freMTPL2.R — freMTPL2freq 特征工程与多划分生成
# 对应路线图：R5（去掉 raw+log 冗余；类别变量 GLM 用 factor、深度模型用 embedding）
# 数据来源：优先 CASdatasets 包；否则读取 data/raw/freMTPL2freq.csv
# =============================================================================

source("00_config.R")

# ---- 加载 ------------------------------------------------------------------
load_fremtpl2 <- function(cfg = CFG) {
  if (requireNamespace("CASdatasets", quietly = TRUE)) {
    data("freMTPL2freq", package = "CASdatasets", envir = environment())
    df <- freMTPL2freq
  } else {
    f <- file.path(cfg$paths$raw, "freMTPL2freq.csv")
    if (!file.exists(f))
      stop("未找到数据：请安装 CASdatasets 包，或将 freMTPL2freq.csv 放入 data/raw/")
    df <- read.csv(f, stringsAsFactors = TRUE)
  }
  as.data.frame(df)
}

# ---- 特征工程 ---------------------------------------------------------------
# 原则（审查报告 §2.4 / 路线图 R1、R5）：
#   1. 每个信息只保留一种表示：Density 只保留 log 版，其余保留原始版；
#   2. 类别变量保持 factor，GLM 端用虚拟变量，深度端用 embedding；
#   3. Exposure 只作 offset，绝不进入特征。
prep_fremtpl2 <- function(df) {
  stopifnot(all(c("ClaimNb", "Exposure") %in% names(df)))
  # Density 存在 0 值（无人区），log(0)=-Inf 会污染标准化；截断到 >=1（审校 §二.2 修复）
  df$Density_log <- log(pmax(df$Density, 1))

  num_features <- c("VehPower", "VehAge", "DrivAge", "BonusMalus", "Density_log")
  cat_features <- c("Area", "VehBrand", "VehGas", "Region")
  for (v in cat_features) df[[v]] <- as.factor(df[[v]])

  cat_levels <- setNames(lapply(cat_features, function(v) nlevels(df[[v]])), cat_features)
  list(df = df, num_features = num_features, cat_features = cat_features,
       cat_levels = cat_levels, task = "poisson",
       target = "ClaimNb", exposure = "Exposure")
}

# ---- 多划分：5 份不同的 70/15/15 ---------------------------------------------
# 修正 Python 版"3 个种子共享同一份划分"的伪重复问题。
make_splits <- function(n, n_splits = CFG$n_splits, seed_base = CFG$seed_base,
                        ratio = CFG$split_ratio) {
  lapply(seq_len(n_splits), function(s) {
    set.seed(seed_base + 1000L * s)
    idx <- sample.int(n)
    n_tr <- floor(ratio["train"] * n); n_va <- floor(ratio["val"] * n)
    list(split = s,
         train = idx[seq_len(n_tr)],
         val   = idx[n_tr + seq_len(n_va)],
         test  = idx[n_tr + n_va + seq_len(n - n_tr - n_va)])
  })
}

# ---- 组装建模数据（数值标准化只用训练集统计量——避免泄漏） -------------------
build_data <- function(prep, idx, scale = NULL) {
  df <- prep$df
  x_num_raw <- as.matrix(df[idx, prep$num_features, drop = FALSE])
  if (is.null(scale)) {
    scale <- list(center = colMeans(x_num_raw),
                  sd     = apply(x_num_raw, 2, sd) + 1e-8)
  }
  x_num <- scale(x_num_raw, center = scale$center, scale = scale$sd)
  # 单类别特征时 sapply 退化为向量；显式构造矩阵保护列数与行数（审校 §二.9）
  x_cat <- vapply(prep$cat_features, function(v) as.integer(df[[v]][idx]),
                  integer(length(idx)))
  if (is.null(dim(x_cat))) x_cat <- matrix(x_cat, nrow = length(idx))
  colnames(x_cat) <- prep$cat_features
  list(x_num = x_num, x_cat = x_cat,
       y = as.numeric(df[[prep$target]][idx]),
       exposure = as.numeric(df[[prep$exposure]][idx]),
       scale = scale)
}

# ---- 一键准备并保存 -----------------------------------------------------------
prep_fremtpl2_all <- function(cfg = CFG, force = FALSE) {
  out <- file.path(cfg$paths$splits, "fremtpl2.rds")
  if (file.exists(out) && !force) { log_msg("freMTPL2 已准备：%s", out); return(invisible(readRDS(out))) }
  prep <- prep_fremtpl2(load_fremtpl2(cfg))
  splits <- make_splits(nrow(prep$df), cfg$n_splits, cfg$seed_base, cfg$split_ratio)
  obj <- list(prep = prep, splits = splits)
  saveRDS(obj, out)
  log_msg("freMTPL2 就绪：%d 行，%d 份划分 -> %s", nrow(prep$df), length(splits), out)
  invisible(obj)
}

if (sys.nframe() == 0L) { init_dirs(); prep_fremtpl2_all() }
