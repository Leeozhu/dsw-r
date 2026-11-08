# =============================================================================
# 06_run_experiments.R — 实验编排（研究方案 §6）
#   E1 光谱；E2 失效注入 2x2；E4 Porto；TCT 严格消融
# 校准预测落独立 rds；threshold 写入 rds；DOUBLECOMP 跑全划分；
# pos_weight/isotonic 逐划分独立拟合
# =============================================================================

source("00_config.R")
source("01_prep_freMTPL2.R")
source("02_prep_porto.R")
source("05_models_torch.R")

save_run <- function(dataset, model, split, seed, fit, d_te, prep, cfg,
                     optimizer = "adamw", threshold = NULL, extra = list(),
                     device = get_device()) {
  tag <- sprintf("%s_%s_split%d_seed%d", dataset, model, split, seed)
  thr <- if (is.null(threshold)) 0.5 else threshold
  pred <- predict_deep(fit$model, d_te, prep, device)
  m <- eval_task(prep$task, d_te$y, pred, d_te$exposure, thr)
  saveRDS(c(list(dataset = dataset, model = model, split = split, seed = seed,
                 y = d_te$y, exposure = d_te$exposure, pred = pred,
                 threshold = thr), extra),
          file.path(cfg$paths$preds, paste0(tag, ".rds")))
  save_weights_rds(fit$model, file.path(cfg$paths$weights, paste0(tag, ".rds")))
  save_json(c(list(model = model, split = split, seed = seed,
                   optimizer = optimizer, lr = cfg$lr, weight_decay = cfg$weight_decay,
                   epochs_run = fit$epochs_run, best_val = fit$best_val,
                   threshold = thr), extra),
            file.path(cfg$paths$weights, paste0(tag, "_meta.json")))
  append_metrics(c(dataset = dataset, model = model, split = split, seed = seed,
                   optimizer = optimizer, epochs = fit$epochs_run, as.list(m)),
                 file.path(cfg$paths$metrics, "metrics_long.csv"))
  log_msg("%s（%s）: %s", tag, optimizer,
          paste(sprintf("%s=%.4f", names(m), m), collapse = "  "))
  invisible(m)
}

# ---- E1：光谱深度层 -----------------------------------------------------------
run_e1_spectrum <- function(cfg = CFG,
                            models = c("MLP_S", "MLP_L", "FTTransformer", "TCTParallel")) {
  obj <- readRDS(file.path(cfg$paths$splits, "fremtpl2.rds"))
  prep <- obj$prep; device <- get_device()
  for (sp in obj$splits) {
    s <- sp$split
    d_tr <- build_data(prep, sp$train)
    d_va <- build_data(prep, sp$val,  d_tr$scale)
    d_te <- build_data(prep, sp$test, d_tr$scale)
    for (mn in models) for (seed in seq_len(cfg$n_seeds)) {
      tag <- sprintf("fremtpl2_%s_split%d_seed%d", mn, s, seed)
      if (file.exists(file.path(cfg$paths$preds, paste0(tag, ".rds")))) next
      fit <- train_deep(mn, d_tr, d_va, prep, cfg, seed = cfg$seed_base + seed,
                        optimizer = "adamw", device = device)
      save_run("fremtpl2", mn, s, seed, fit, d_te, prep, cfg, device = device)
      rm(fit); gc(verbose = FALSE)
    }
  }
  invisible(TRUE)
}

# ---- E2：2x2 失效注入 --------------------------------------------------------
run_e2_failure_injection <- function(cfg = CFG) {
  obj <- readRDS(file.path(cfg$paths$splits, "fremtpl2.rds"))
  prep <- obj$prep; device <- get_device()
  cells <- list(
    A = list(model = "FTTransformer", optimizer = "adamw"),
    B = list(model = "FTTransformer", optimizer = "adam"),
    C = list(model = "SeqOneTrans",   optimizer = "adamw"),
    D = list(model = "SeqOneTrans",   optimizer = "adam")
  )
  for (sp in obj$splits[seq_len(min(cfg$e2_splits, length(obj$splits)))]) {
    s <- sp$split
    d_tr <- build_data(prep, sp$train)
    d_va <- build_data(prep, sp$val,  d_tr$scale)
    d_te <- build_data(prep, sp$test, d_tr$scale)
    for (cell_id in names(cells)) for (seed in seq_len(cfg$e2_seeds)) {
      cell <- cells[[cell_id]]
      run_model <- sprintf("E2%s_%s_%s", cell_id, cell$model, cell$optimizer)
      tag <- sprintf("fremtpl2_%s_split%d_seed%d", run_model, s, seed)
      if (file.exists(file.path(cfg$paths$preds, paste0(tag, ".rds")))) next
      fit <- train_deep(cell$model, d_tr, d_va, prep, cfg,
                        seed = cfg$seed_base + seed, optimizer = cell$optimizer,
                        device = device, verbose = FALSE)
      ssave_run("fremtpl2", run_model, s, seed, fit, d_te, prep, cfg,
                optimizer = cell$optimizer, device = device,
                extra = list(e2_cell = cell_id, base_model = cell$model))
      rm(fit); gc(verbose = FALSE)
    }
  }
  invisible(TRUE)
}

# ---- TCT 严格消融：变体与完整模型共享类/形状/超参/种子，仅前向屏蔽分支 ----------
run_ablation_fremtpl2 <- function(cfg = CFG,
                                  variants = c(full = "full", no_attn = "tcn",
                                               no_tcn = "attn", avg_fusion = "avg")) {
  obj <- readRDS(file.path(cfg$paths$splits, "fremtpl2.rds"))
  prep <- obj$prep; device <- get_device()
  for (sp in obj$splits) {
    s <- sp$split
    d_tr <- build_data(prep, sp$train)
    d_va <- build_data(prep, sp$val,  d_tr$scale)
    d_te <- build_data(prep, sp$test, d_tr$scale)
    for (vn in names(variants)) for (seed in seq_len(cfg$n_seeds)) {
      br <- variants[[vn]]
      run_model <- if (vn == "full") "TCTParallel" else sprintf("TCT_abl_%s", vn)
      tag <- sprintf("fremtpl2_%s_split%d_seed%d", run_model, s, seed)
      if (file.exists(file.path(cfg$paths$preds, paste0(tag, ".rds")))) next
      fit <- train_deep("TCTParallel", d_tr, d_va, prep, cfg, branch = br,
                        seed = cfg$seed_base + seed, device = device, verbose = FALSE)
      save_run("fremtpl2", run_model, s, seed, fit, d_te, prep, cfg,
               device = device, extra = list(branch = br))
      rm(fit); gc(verbose = FALSE)
    }
  }
  invisible(TRUE)
}

# ---- E4：Porto 案例 ----------------------------------------------------------
# (a) 失效管线复现：重采样 x pos_weight=10，全划分跑（seed1）
# (b) 纠正管线：pos_weight 与 isotonic 在【本划分验证集】独立拟合
run_e4_porto <- function(cfg = CFG, models = c("MLP_S", "FTTransformer")) {
  obj <- readRDS(file.path(cfg$paths$splits, "porto.rds"))
  prep <- obj$prep; device <- get_device()

  for (sp in obj$splits) {
    s <- sp$split
    d_tr <- build_data_bin(prep, sp$train)
    d_va <- build_data_bin(prep, sp$val, d_tr$scale)
    d_te <- build_data_bin(prep, sp$test, d_tr$scale)

    # ---- (a) 失效管线复现 ----------------------------------------------------
    tag_bad <- sprintf("porto_MLP_S_DOUBLECOMP_split%d_seed1", s)
    if (!file.exists(file.path(cfg$paths$preds, paste0(tag_bad, ".rds")))) {
      pos_idx <- which(d_tr$y == 1); neg_idx <- which(d_tr$y == 0)
      set.seed(cfg$seed_base + s)                      # 每划分独立重采样
      bal_idx <- c(sample(pos_idx, length(neg_idx), replace = TRUE), neg_idx)
      d_tr_bal <- list(x_num = d_tr$x_num[bal_idx, , drop = FALSE],
                       x_cat = d_tr$x_cat[bal_idx, , drop = FALSE],
                       y = d_tr$y[bal_idx], exposure = NULL, scale = d_tr$scale)
      fit_bad <- train_deep("MLP_S", d_tr_bal, d_va, prep, cfg, pos_weight = 10,
                            seed = cfg$seed_base + 1, optimizer = "adamw",
                            device = device, verbose = FALSE)
      save_run("porto", "MLP_S_DOUBLECOMP", s, 1L, fit_bad, d_te, prep, cfg, device = device,
               extra = list(note = "failure reproduction: resample x pos_weight=10"))
      rm(fit_bad); gc(verbose = FALSE)
    }

    # ---- (b) 纠正管线：逐划分独立选 pos_weight + 阈值 + isotonic ----------------
    best_auc <- -Inf; best_w <- 1; best_p <- NULL
    for (w in cfg$porto_pos_weight_grid) {
      fit <- train_deep("MLP_S", d_tr, d_va, prep, cfg, pos_weight = w,
                        seed = cfg$seed_base + 1, device = device, verbose = FALSE)
      p <- predict_deep(fit$model, d_va, prep, device)
      a <- auc_binary(d_va$y, p)
      if (a > best_auc) { best_auc <- a; best_w <- w; best_p <- p }
      rm(fit); gc(verbose = FALSE)
    }
    grid <- seq(0.01, 0.99, by = 0.01)
    thr <- grid[which.max(sapply(grid, function(th) cls_metrics(d_va$y, best_p, th)$f1))]
    iso <- stats::isoreg(best_p, d_va$y)
    ord <- order(iso$x)
    cal_fun <- approxfun(iso$x[ord], iso$yf[ord], method = "constant", rule = 2)
    save_json(list(split = s, pos_weight = best_w, threshold = thr,
                   note = "per-split single compensation + F1 threshold + isotonic"),
              file.path(cfg$paths$metrics, sprintf("porto_correction_split%d.json", s)))

    for (mn in models) for (seed in seq_len(cfg$n_seeds)) {
      tag <- sprintf("porto_%s_split%d_seed%d", mn, s, seed)
      if (file.exists(file.path(cfg$paths$preds, paste0(tag, ".rds")))) next
      fit <- train_deep(mn, d_tr, d_va, prep, cfg, pos_weight = best_w,
                        seed = cfg$seed_base + seed, optimizer = "adamw",
                        device = device, verbose = FALSE)
      pred_raw <- predict_deep(fit$model, d_te, prep, device)
      pred_cal <- pmin(pmax(cal_fun(pred_raw), 0), 1)

      save_run("porto", mn, s, seed, fit, d_te, prep, cfg, device = device,
               threshold = thr,
               extra = list(pos_weight = best_w, calibrated = FALSE))

      # 校准后预测落独立 rds，供 collect_metrics 重建校准前后对比
      saveRDS(list(dataset = "porto", model = paste0(mn, "_CALIBRATED"),
                   split = s, seed = seed, y = d_te$y, exposure = NULL,
                   pred = pred_cal, threshold = thr, calibrated = TRUE),
              file.path(cfg$paths$preds,
                        sprintf("porto_%s_CALIBRATED_split%d_seed%d.rds", mn, s, seed)))
      log_msg("porto %s split%d seed%d: pos_weight=%g thr=%.2f",
              mn, s, seed, best_w, thr)
      rm(fit); gc(verbose = FALSE)
    }
  }
  invisible(TRUE)
}

if (sys.nframe() == 0L) {
  init_dirs()
  run_e1_spectrum()
  run_e2_failure_injection()
  run_ablation_fremtpl2()
}
