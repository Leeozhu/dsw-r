# =============================================================================
# 02_prep_porto.R — Porto Seguro 预处理
# 对应路线图：R6（缺失值 -1 不用中位数填充，保留信号 + 显式缺失指示列）
# 数据来源：Kaggle porto-seguro-safe-driver-prediction 的 train.csv -> data/raw/porto_train.csv
# =============================================================================

source("00_config.R")

load_porto <- function(cfg = CFG) {
  f <- file.path("data", "raw", "porto_train.csv")
  if (!file.exists(f))
    stop("请将 Kaggle Porto Seguro 的 train.csv 重命名为 porto_train.csv 放入 data/raw/")
  read.csv(f, stringsAsFactors = FALSE)
}

# 处理约定：
#  - 列名以 _cat / _bin 结尾 -> 类别；其余 ps_* 数值。
#  - -1 表示缺失：类别列把 -1 变为独立水平 "MISSING"；数值列保留 -1，
#    并对缺失率 > 0 的列增加 was_missing_<列名> 指示变量。
#  - 不做任何重采样；不平衡补偿只在训练端用单一 pos_weight（见 05）。
prep_porto <- function(df) {
  df <- df[, setdiff(names(df), "id")]
  feat <- setdiff(names(df), "target")
  cat_features <- grep("(_cat|_bin)$", feat, value = TRUE)
  num_features <- setdiff(feat, cat_features)

  for (v in cat_features) {
    x <- df[[v]]
    x[x == -1] <- max(x, na.rm = TRUE) + 1L     # 缺失成为独立水平
    df[[v]] <- factor(x)
  }
  for (v in num_features) {
    if (any(df[[v]] == -1)) {
      df[[paste0("was_missing_", v)]] <- as.integer(df[[v]] == -1)
    }
  }
  num_features <- c(num_features, grep("^was_missing_", names(df), value = TRUE))
  cat_levels <- setNames(lapply(cat_features, function(v) nlevels(df[[v]])), cat_features)

  list(df = df, num_features = num_features, cat_features = cat_features,
       cat_levels = cat_levels, task = "binary", target = "target", exposure = NULL)
}

build_data_bin <- function(prep, idx, scale = NULL) {
  df <- prep$df
  x_num_raw <- as.matrix(df[idx, prep$num_features, drop = FALSE])
  storage.mode(x_num_raw) <- "double"
  if (is.null(scale))
    scale <- list(center = colMeans(x_num_raw), sd = apply(x_num_raw, 2, sd) + 1e-8)
  x_num <- scale(x_num_raw, scale$center, scale$sd)
  x_cat <- vapply(prep$cat_features, function(v) as.integer(df[[v]][idx]),
                  integer(length(idx)))
  if (is.null(dim(x_cat))) x_cat <- matrix(x_cat, nrow = length(idx))
  colnames(x_cat) <- prep$cat_features
  list(x_num = x_num, x_cat = x_cat,
       y = as.numeric(df[[prep$target]][idx]), exposure = NULL, scale = scale)
}

prep_porto_all <- function(cfg = CFG, force = FALSE) {
  out <- file.path(cfg$paths$splits, "porto.rds")
  if (file.exists(out) && !force) { log_msg("Porto 已准备：%s", out); return(invisible(readRDS(out))) }
  prep <- prep_porto(load_porto(cfg))
  splits <- make_splits(nrow(prep$df), cfg$n_splits, cfg$seed_base, cfg$split_ratio)
  obj <- list(prep = prep, splits = splits)
  saveRDS(obj, out)
  log_msg("Porto 就绪：%d 行，正类占比 %.3f，%d 份划分",
          nrow(prep$df), mean(prep$df$target), length(splits))
  invisible(obj)
}

if (sys.nframe() == 0L) { init_dirs(); prep_porto_all() }
