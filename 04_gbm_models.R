# =============================================================================
# 04_gbm_models.R — XGBoost（透明度光谱 L2：集成不透明）
#  - freMTPL2：objective=count:poisson，base_margin=log(Exposure)（频率+offset）
#  - Porto：scale_pos_weight=1，与深度模型一样只靠阈值调优（单一补偿口径可比）
#  - 每个 split x seed 运行，与深度模型形成配对结构
# =============================================================================

source("00_config.R")
library(xgboost)

make_mm <- function(prep, df_sub, colnames_ref = NULL) {
  # df_sub 只含特征列；直接对特征全集做 model.matrix（勿引用 target/exposure，
  # 它们不在子集列中，会报 variable not found——审校 §二.4 修复）
  mm <- model.matrix(~ ., data = df_sub[, c(prep$num_features, prep$cat_features),
                                        drop = FALSE])[, -1, drop = FALSE]
  if (!is.null(colnames_ref)) {
    for (mc in setdiff(colnames_ref, colnames(mm)))
      mm <- cbind(mm, setNames(data.frame(rep(0, nrow(mm))), mc))
    mm <- mm[, colnames_ref, drop = FALSE]
  }
  mm
}

run_xgb_dataset <- function(dataset = c("fremtpl2", "porto"), cfg = CFG) {
  dataset <- match.arg(dataset)
  obj <- readRDS(file.path(cfg$paths$splits, paste0(dataset, ".rds")))
  prep <- obj$prep; df <- prep$df

  for (sp in obj$splits) for (seed in seq_len(cfg$n_seeds)) {
    s <- sp$split
    tag <- sprintf("%s_XGBoost_split%d_seed%d", dataset, s, seed)
    if (file.exists(file.path(cfg$paths$preds, paste0(tag, ".rds")))) next
    set_all_seeds(cfg$seed_base + seed)
    mm_tr <- make_mm(prep, df[sp$train, ])
    mm_va <- make_mm(prep, df[sp$val, ], colnames(mm_tr))
    mm_te <- make_mm(prep, df[sp$test, ], colnames(mm_tr))

    if (prep$task == "poisson") {
      dtrain <- xgb.DMatrix(mm_tr, label = df$ClaimNb[sp$train],
                            base_margin = log(df$Exposure[sp$train]))
      dval   <- xgb.DMatrix(mm_va, label = df$ClaimNb[sp$val],
                            base_margin = log(df$Exposure[sp$val]))
      dtest  <- xgb.DMatrix(mm_te, base_margin = log(df$Exposure[sp$test]))
      params <- list(objective = "count:poisson", eta = 0.05, max_depth = 7,
                     subsample = 0.8, colsample_bytree = 0.8, lambda = 1)
      y_te <- df$ClaimNb[sp$test]; e_te <- df$Exposure[sp$test]
    } else {
      dtrain <- xgb.DMatrix(mm_tr, label = df$target[sp$train])
      dval   <- xgb.DMatrix(mm_va, label = df$target[sp$val])
      dtest  <- xgb.DMatrix(mm_te)
      params <- list(objective = "binary:logistic", eta = 0.05, max_depth = 6,
                     subsample = 0.8, colsample_bytree = 0.8, scale_pos_weight = 1)
      y_te <- df$target[sp$test]; e_te <- NULL
    }

    bst <- xgb.train(params, dtrain, nrounds = cfg$xgb_nrounds,
                     evals = list(val = dval),
                     early_stopping_rounds = cfg$xgb_patience, verbose = 0)

    pred <- if (prep$task == "poisson") as.numeric(predict(bst, dtest)) / e_te
            else as.numeric(predict(bst, dtest))
    m <- eval_task(prep$task, y_te, pred, e_te)
    # 早停未触发时 best_iteration 为 NULL，需保护避免 c() 丢字段（审校 §二.9）
    best_iter <- if (is.null(bst$best_iteration)) NA_integer_ else bst$best_iteration
    saveRDS(list(dataset = dataset, model = "XGBoost", split = s, seed = seed,
                 y = y_te, exposure = e_te, pred = pred,
                 best_iteration = best_iter),
            file.path(cfg$paths$preds, paste0(tag, ".rds")))
    append_metrics(c(dataset = dataset, model = "XGBoost", split = s, seed = seed,
                     best_iter = best_iter, as.list(m)),
                   file.path(cfg$paths$metrics, "metrics_long.csv"))
    log_msg("%s: %s  best_iter=%d/%d", tag,
        paste(sprintf("%s=%.4f", names(m), m), collapse = "  "),
        best_iter, cfg$xgb_nrounds)
  }
  invisible(TRUE)
}

if (sys.nframe() == 0L) { init_dirs(); run_xgb_dataset("fremtpl2") }
