# =============================================================================
# 00_config.R — 全局配置、指标函数、通用工具
# 项目：透明度光谱与模型可验证性（研究方案 v1.0）
# 组织原则：模型按透明度层级（L0-L4）注册；失效注入为 2x2 因子设计。
# 约定：所有进论文的数字必须由脚本写入 outputs/，禁止手工转录。
# =============================================================================

CFG <- list(
  seed_base   = 42L,
  n_splits    = 5L,      # 5 份不同 70/15/15 划分（替代旧版伪重复 3 种子）
  n_seeds     = 3L,
  split_ratio = c(train = 0.70, val = 0.15, test = 0.15),
  # 深度模型训练
  epochs       = 200L,
  patience     = 20L,
  batch_size   = 1024L,
  lr           = 1e-3,
  weight_decay = 1e-4,   # 主实验一律 AdamW（解耦）；耦合 Adam 仅用于失效注入
  d_model      = 64L,
  n_heads      = 4L,
  n_layers     = 2L,
  dropout      = 0.1,
  # E2 失效注入：2x2 = 输入表示 {tokenizer, seq1} x 优化器 {adamw, adam}
  e2_splits    = 3L,     # E2 可用较少划分控制算力
  e2_seeds     = 2L,
  # Porto
  porto_pos_weight_grid = c(1, 2, 3, 5),
  # xgboost
  xgb_nrounds  = 2000L,
  xgb_patience = 50L,
  # bootstrap CI
  n_boot       = 500L,
  paths = list(raw = "data/raw", splits = "data/splits",
               metrics = "outputs/metrics", preds = "outputs/metrics/preds",
               weights = "outputs/weights", figures = "outputs/figures")
)

# 透明度光谱注册表：level 用于 RQ1 汇总与 RQ3 清单分档
MODEL_SPECTRUM <- data.frame(
  model = c("GLM", "GAM", "XGBoost", "MLP_S", "MLP_L", "FTTransformer", "TCTParallel"),
  level = c("L0", "L1", "L2", "L3", "L4", "L4", "L4"),
  label = c("GLM（完全透明）", "GAM（半透明）", "XGBoost（集成不透明）",
            "MLP-small（深度不透明·小）", "MLP-large（深度不透明·大）",
            "FT-Transformer（深度不透明·注意力）", "TCT-Parallel（深度不透明·混合）"),
  stringsAsFactors = FALSE
)

init_dirs <- function(cfg = CFG) {
  for (p in cfg$paths) dir.create(p, recursive = TRUE, showWarnings = FALSE)
  invisible(TRUE)
}

set_all_seeds <- function(seed) {
  set.seed(seed)
  if (requireNamespace("torch", quietly = TRUE)) torch::torch_manual_seed(seed)
  invisible(seed)
}

get_device <- function() {
  if (!requireNamespace("torch", quietly = TRUE)) stop("需要安装 torch 包")
  dev <- tryCatch({
    if (torch::cuda_is_available()) torch::torch_device("cuda")
    else if (isTRUE(tryCatch(torch::backends_mps_is_available(), error = function(e) FALSE)))
      torch::torch_device("mps")
    else torch::torch_device("cpu")
  }, error = function(e) torch::torch_device("cpu"))
  message(sprintf("[device] 使用 %s", dev$type))
  dev
}

# ---- 写盘工具：一切数字自动落盘 ---------------------------------------------
save_json <- function(obj, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  jsonlite::write_json(obj, path, auto_unbox = TRUE, digits = 10, pretty = TRUE)
  invisible(path)
}

# ---- 写盘工具：append_metrics 改为"读—合并—重写"，列错位免疫 -----------------
append_metrics <- function(row, file) {
  dir.create(dirname(file), recursive = TRUE, showWarnings = FALSE)
  new_row <- as.data.frame(row)
  if (file.exists(file)) {
    old <- tryCatch(read.csv(file, stringsAsFactors = FALSE),
                    error = function(e) NULL)
    if (!is.null(old)) {
      all_cols <- union(names(old), names(new_row))
      for (cc in setdiff(all_cols, names(old))) old[[cc]] <- NA
      for (cc in setdiff(all_cols, names(new_row))) new_row[[cc]] <- NA
      merged <- rbind(old[, all_cols, drop = FALSE],
                      new_row[, all_cols, drop = FALSE])
      write.csv(merged, file, row.names = FALSE)
      return(invisible(file))
    }
  }
  write.csv(new_row, file, row.names = FALSE)
  invisible(file)
}

log_msg <- function(fmt, ...) {
  cat(sprintf("[%s] %s\n", format(Sys.time(), "%H:%M:%S"), sprintf(fmt, ...)))
}

# =============================================================================
# 指标
# =============================================================================

# Poisson deviance（每条保单平均）：y 索赔次数，mu 期望计数（rate x exposure）
poisson_deviance <- function(y, mu) {
  mu <- pmax(mu, 1e-10)
  term <- ifelse(y == 0, mu, y * log(y / mu) - (y - mu))
  2 * mean(term)
}

# Gini：按【频率】排序，暴露度仅作累积权重（修正旧版按期望计数排序的口径）
gini_freq <- function(y_count, exposure, pred_rate) {
  ord <- order(pred_rate, decreasing = TRUE)
  y <- y_count[ord]; e <- exposure[ord]
  ce <- c(0, cumsum(e) / sum(e)); cy <- c(0, cumsum(y) / sum(y))
  area <- sum(diff(ce) * (head(cy, -1) + tail(cy, -1)) / 2)
  2 * area - 1                                   # 完美→+1，随机→0，反向→−1
}

# 校准性：按预测频率分 10 组，比较组内预测均值与实际频率（返回 log 比率向量的 RMSE）
calibration_rmse <- function(y, exposure, pred_rate, n_groups = 10L) {
  grp <- cut(rank(pred_rate) / length(pred_rate), breaks = seq(0, 1, length.out = n_groups + 1),
             include.lowest = TRUE, labels = FALSE)
  pred_mu <- tapply(pred_rate * exposure, grp, sum)
  obs_mu  <- tapply(y, grp, sum)
  exp_tot <- tapply(exposure, grp, sum)
  ok <- exp_tot > 0
  sqrt(mean((log(pmax(pred_mu[ok], 1e-10)) - log(pmax(obs_mu[ok], 1e-10)))^2))
}

auc_binary <- function(y, p) {
  r <- rank(p, ties.method = "average")
  n1 <- sum(y == 1); n0 <- sum(y == 0)
  (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

brier_score <- function(y, p) mean((y - p)^2)

mae_score <- function(y, p) mean(abs(y - p))
rmse_score <- function(y, p) sqrt(mean((y - p)^2))
r2_score <- function(y, p) 1 - sum((y - p)^2) / sum((y - mean(y))^2)

cls_metrics <- function(y, p, thr = 0.5) {
  pred <- as.integer(p >= thr)
  tp <- sum(pred == 1 & y == 1); fp <- sum(pred == 1 & y == 0)
  fn <- sum(pred == 0 & y == 1)
  list(threshold = thr, recall = tp / max(tp + fn, 1),
       precision = tp / max(tp + fp, 1), f1 = 2 * tp / max(2 * tp + fp + fn, 1))
}

eval_task <- function(task, y, pred, exposure = NULL, thr = 0.5) {
  if (task == "poisson") {
    mu <- pred * exposure
    c(poisson_deviance = poisson_deviance(y, mu),
      gini_freq        = gini_freq(y, exposure, pred),
      calib_rmse       = calibration_rmse(y, exposure, pred))
  } else if (task == "binary") {
    m <- cls_metrics(y, pred, thr)
    c(auc = auc_binary(y, pred), brier = brier_score(y, pred), unlist(m))
  } else {
    c(mae = mae_score(y, pred), rmse = rmse_score(y, pred), r2 = r2_score(y, pred))
  }
}

boot_ci <- function(stat_fun, B = CFG$n_boot, seed = 1L) {
  set.seed(seed)
  vals <- replicate(B, stat_fun())
  c(mean = mean(vals), lo = unname(quantile(vals, 0.025)),
    hi = unname(quantile(vals, 0.975)))
}
