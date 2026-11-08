# =============================================================================
# 04a_gam_models.R — GAM/GA2M（透明度光谱 L1：半透明）
# 
# 核心改进：
#   1. 仅保留一维主效应 s()，不使用 ti()/te() —— 彻底解决交互项吞噬主效应
#      导致曲线全平的问题，保证监管可解释性。
#   2. 使用 bam(method="fREML", discrete=TRUE) 替代 gam(method="REML")。
#      freMTPL2 68 万行数据拟合时间从 2.5 小时缩短至 10-20 分钟。
#   3. 内置 gam.check() 和 concurvity() 诊断，写入 txt 供审计证据使用。
#   4. 效应曲线提取用 pdf(NULL) 抑制画图，仅保留一维 s() 主效应。
# =============================================================================

source("00_config.R")
source("01_prep_freMTPL2.R")

# 纯主效应公式：offset 必须写进公式，确保 predict(newdata) 自动携带
gam_formula_freq <- as.formula(
  ClaimNb ~ s(BonusMalus, k = 10) + s(DrivAge, k = 10) + s(VehAge, k = 10) +
    s(VehPower, k = 6) + s(Density_log, k = 6) +
    Area + VehBrand + VehGas + Region + offset(log(Exposure))
)

# ---- 效应曲线：null device 抑制绘图，只取返回值 ----------------------------
extract_gam_curves <- function(fit, tag) {
  pd <- NULL
  grDevices::pdf(NULL)
  pd <- tryCatch(
    mgcv::plot.gam(fit, pages = 1, se = TRUE),
    error = function(e) NULL
  )
  tryCatch(grDevices::dev.off(), error = function(e) NULL)
  
  if (is.null(pd) || !length(pd)) return(NULL)
  
  curve <- lapply(seq_along(pd), function(i) {
    el <- pd[[i]]
    # 确保是一维平滑项
    if (is.list(el$x) || is.null(el$fit) || length(el$x) != length(el$fit)) return(NULL)
    term_lab <- if (!is.null(el$xlab)) el$xlab
    else if (!is.null(names(pd)[i])) names(pd)[i]
    else paste0("smooth_", i)
    data.frame(term = as.character(term_lab),
               x    = as.numeric(el$x),
               fit  = as.numeric(el$fit),
               se   = if (!is.null(el$se) && length(el$se) == length(el$fit))
                 as.numeric(el$se) else NA_real_,
               stringsAsFactors = FALSE)
  })
  curve <- Filter(Negate(is.null), curve)
  if (!length(curve)) return(NULL)
  do.call(rbind, curve)
}

# ---- 主流程 -----------------------------------------------------------------
run_gam_fremtpl2 <- function(cfg = CFG, use_bam = TRUE) {
  if (!requireNamespace("mgcv", quietly = TRUE))
    stop("需要 mgcv 包（install.packages('mgcv')）")
  
  obj <- readRDS(file.path(cfg$paths$splits, "fremtpl2.rds"))
  prep <- obj$prep; df <- prep$df
  
  for (sp in obj$splits) {
    s <- sp$split
    tag     <- sprintf("fremtpl2_GAM_split%d_seed0", s)
    pred_f  <- file.path(cfg$paths$preds,   paste0(tag, ".rds"))
    curve_f <- file.path(cfg$paths$metrics, sprintf("gam_effects_split%d.csv", s))
    diag_f  <- file.path(cfg$paths$metrics, sprintf("gam_diagnostics_split%d.txt", s))
    
    if (file.exists(pred_f) && file.exists(curve_f) && file.exists(diag_f)) {
      log_msg("跳过已完成：%s（pred + curve + diag 均存在）", tag)
      next
    }
    
    d_tr <- df[sp$train, ]; d_te <- df[sp$test, ]
    
    t0 <- Sys.time()
    fit <- if (use_bam) {
      mgcv::bam(gam_formula_freq, family = poisson(), data = d_tr,
                method = "fREML", discrete = TRUE)
    } else {
      mgcv::gam(gam_formula_freq, family = poisson(), data = d_tr,
                method = "REML")
    }
    fit_sec <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    
    # 诊断写盘（k 充分性 + 共线性）
    sink(diag_f)
    cat("=== gam.check ===\n"); print(mgcv::gam.check(fit, rep = 200))
    cat("\n=== concurvity ===\n"); print(round(mgcv::concurvity(fit, full = FALSE)$estimate, 3))
    cat(sprintf("\nFit time: %.1f seconds\n", fit_sec))
    sink()
    
    # 预测：offset 在公式中，自动携带
    mu_hat   <- as.numeric(predict(fit, newdata = d_te, type = "response"))
    rate_hat <- mu_hat / d_te$Exposure
    
    # 指标计算与写盘
    m <- eval_task("poisson", d_te$ClaimNb, rate_hat, d_te$Exposure)
    saveRDS(list(dataset = "fremtpl2", model = "GAM", split = s, seed = 0L,
                 y = d_te$ClaimNb, exposure = d_te$Exposure, pred = rate_hat),
            pred_f)
    
    # 效应曲线写盘
    curve_df <- extract_gam_curves(fit, tag)
    if (!is.null(curve_df)) {
      write.csv(curve_df, curve_f, row.names = FALSE)
    }
    
    append_metrics(c(dataset = "fremtpl2", model = "GAM", split = s, seed = 0L,
                     as.list(m)), file.path(cfg$paths$metrics, "metrics_long.csv"))
    log_msg("GAM split %d: PD=%.4f  Gini=%.4f  calib=%.4f  fit=%.1fs  (%s)",
            s, m["poisson_deviance"], m["gini_freq"], m["calib_rmse"],
            fit_sec, if (use_bam) "bam" else "gam")
  }
  invisible(TRUE)
}

if (sys.nframe() == 0L) { init_dirs(); run_gam_fremtpl2() }