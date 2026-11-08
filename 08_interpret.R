# =============================================================================
# 08_interpret.R — 可解释性：置换重要性 + 精算一致性（如实口径）
# 对应路线图：R7（全文统一称"置换重要性"；方向一致率如实报告；
#              删除"驾驶员年龄单调正效应"类未验证表述）
# =============================================================================

source("00_config.R")
source("01_prep_freMTPL2.R")
source("05_models_torch.R")

# ---- 置换重要性：打乱测试集某特征 -> Poisson deviance 增量 --------------------
perm_importance_torch <- function(model, d_te, prep, device, n_rep = 5L, seed = 7L) {
  t <- to_tensors(d_te, device)
  base_pd <- with_no_grad({
    out <- as_array(model(t$x_num, t$x_cat)$cpu())
    poisson_deviance(d_te$y, exp(out) * d_te$exposure)
  })
  set.seed(seed)
  res <- list()

  # 在 R 数组层面打乱后重建张量，避免 torch 张量切片赋值 [<-.torch_tensor 的
  # 形状/类型限制（审校 §二.6 修复：xn[, j] <- ... 在 R torch 下会报错）
  n_te <- nrow(d_te$x_num)
  for (j in seq_along(prep$num_features)) {
    delta <- replicate(n_rep, {
      x_arr <- as_array(t$x_num$cpu())
      x_arr[, j] <- x_arr[sample.int(n_te), j]
      xn <- torch_tensor(x_arr, dtype = torch_float(), device = device)
      with_no_grad({
        out <- as_array(model(xn, t$x_cat)$cpu())
        poisson_deviance(d_te$y, exp(out) * d_te$exposure)
      }) - base_pd
    })
    res[[prep$num_features[j]]] <- c(mean = mean(delta), sd = sd(delta))
  }
  for (j in seq_along(prep$cat_features)) {
    delta <- replicate(n_rep, {
      x_arr <- as_array(t$x_cat$cpu())
      x_arr[, j] <- x_arr[sample.int(n_te), j]
      xc <- torch_tensor(x_arr, dtype = torch_long(), device = device)
      with_no_grad({
        out <- as_array(model(t$x_num, xc)$cpu())
        poisson_deviance(d_te$y, exp(out) * d_te$exposure)
      }) - base_pd
    })
    res[[prep$cat_features[j]]] <- c(mean = mean(delta), sd = sd(delta))
  }
  do.call(rbind, lapply(names(res), function(nm)
    data.frame(feature = nm, delta_pd = res[[nm]]["mean"], sd = res[[nm]]["sd"])))
}

# ---- 效应方向：数值特征比较"高分位 vs 低分位"的预测频率符号 -------------------
effect_direction <- function(model, d_te, prep, device) {
  t <- to_tensors(d_te, device)
  pred_at <- function(xn) with_no_grad({
    mean(exp(as_array(model(xn, t$x_cat)$cpu())))
  })
  rows <- list()
  x_arr0 <- as_array(t$x_num$cpu())   # 数组层面改列后重建张量（避免张量切片赋值）
  for (j in seq_along(prep$num_features)) {
    nm <- prep$num_features[j]
    x <- d_te$x_num[, j]
    lo <- quantile(x, 0.1); hi <- quantile(x, 0.9)
    a_lo <- x_arr0; a_lo[, j] <- lo
    a_hi <- x_arr0; a_hi[, j] <- hi
    xn_lo <- torch_tensor(a_lo, dtype = torch_float(), device = device)
    xn_hi <- torch_tensor(a_hi, dtype = torch_float(), device = device)
    rows[[nm]] <- data.frame(feature = nm,
                             direction = sign(pred_at(xn_hi) - pred_at(xn_lo)))
  }
  do.call(rbind, rows)
}

# ---- 精算先验表（freMTPL2 车险常识方向） --------------------------------------
actuarial_priors <- function() {
  data.frame(
    feature = c("BonusMalus", "DrivAge", "VehAge", "VehPower", "Density_log"),
    prior_direction = c(+1, -1, -1, +1, +1),   # 经验费率↑、驾驶年龄↑风险↓、车龄↑风险↓
    stringsAsFactors = FALSE
  )
}

# ---- 主流程：对 freMTPL2 split1 上训练好的指定模型执行 ------------------------
run_interpret <- function(model_name = "TCTParallel", split = 1L, seed = 1L, cfg = CFG) {
  obj <- readRDS(file.path(cfg$paths$splits, "fremtpl2.rds"))
  prep <- obj$prep; device <- get_device()
  sp <- obj$splits[[split]]
  d_tr <- build_data(prep, sp$train); d_te <- build_data(prep, sp$test, d_tr$scale)

  # 加载已存权重（06 训练后自动写盘；此处复载，不重训）
  wf <- file.path(cfg$paths$weights,
                  sprintf("fremtpl2_%s_split%d_seed%d.rds", model_name, split, seed))
  if (!file.exists(wf)) stop("未找到权重：", wf, " —— 请先运行 06 主实验")
  model <- build_model(model_name, prep, cfg)
  model$load_state_dict(lapply(readRDS(wf),
    function(a) torch_tensor(a, dtype = torch_float(), device = device)))
  model$eval()

  pi_df <- perm_importance_torch(model, d_te, prep, device)
  write.csv(pi_df, file.path(cfg$paths$metrics,
            sprintf("perm_importance_%s_split%d.csv", model_name, split)), row.names = FALSE)

  dir_df <- effect_direction(model, d_te, prep, device)
  pri <- actuarial_priors()
  cons <- merge(dir_df, pri, by = "feature")
  cons$agree <- cons$direction == cons$prior_direction
  rate <- mean(cons$agree)
  write.csv(cons, file.path(cfg$paths$metrics,
            sprintf("actuarial_consistency_%s_split%d.csv", model_name, split)),
            row.names = FALSE)
  save_json(list(model = model_name, split = split,
                 method = "Permutation Importance",   # 如实用词，不得写作 SHAP
                 direction_agreement_rate = rate,
                 wording_note = sprintf("方向一致率 %.1f%%（部分一致时论文必须如实表述）",
                                        100 * rate)),
            file.path(cfg$paths$metrics,
                      sprintf("actuarial_consistency_%s_split%d.json", model_name, split)))
  log_msg("%s split%d：置换重要性已写盘；方向一致率 %.1f%%", model_name, split, 100 * rate)
  invisible(list(importance = pi_df, consistency = cons, rate = rate))
}

if (sys.nframe() == 0L) { init_dirs(); run_interpret() }
