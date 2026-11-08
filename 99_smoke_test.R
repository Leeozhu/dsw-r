# =============================================================================
# 99_smoke_test.R — 端到端冒烟测试（合成数据，分钟级）
# 验证：指标、数据管线、torch 模型（含 E2 双优化器）、审计、检测矩阵链路。
# 用法：Rscript 99_smoke_test.R
# =============================================================================
message("cwd: ", normalizePath(getwd()))
message("05_models_torch.R: ", normalizePath("05_models_torch.R"))
setwd()
source("00_config.R")
source("09_audit.R")   # 链条带入 01/05
init_dirs()

log_msg("== 1/5 指标 ==")
y <- rpois(2000, 0.15); e <- runif(2000, 0.1, 1)
stopifnot(is.finite(poisson_deviance(y, 0.15 * e)))
stopifnot(abs(gini_freq(y, e, rep(0.15, 2000))) < 0.2)
stopifnot(is.finite(calibration_rmse(y, e, rep(0.15, 2000))))
log_msg("指标 OK")

log_msg("== 2/5 合成数据 ==")
n <- 3000
set.seed(1)
synth <- data.frame(
  ClaimNb = rpois(n, 0.1), Exposure = runif(n, 0.05, 1),
  VehPower = rnorm(n, 6, 2), VehAge = pmax(rnorm(n, 6, 4), 0),
  DrivAge = pmax(rnorm(n, 45, 12), 18), BonusMalus = pmax(rnorm(n, 80, 25), 50),
  Density = rlnorm(n, 8, 1.5),   # prep_fremtpl2 内部自算 Density_log（审校 §二.7 修复）
  Area = factor(sample(LETTERS[1:6], n, TRUE)),
  VehBrand = factor(sample(paste0("B", 1:11), n, TRUE)),
  VehGas = factor(sample(c("Diesel", "Regular"), n, TRUE)),
  Region = factor(sample(paste0("R", 1:22), n, TRUE)))
prep <- prep_fremtpl2(synth)
sp <- make_splits(n, n_splits = 1L)[[1]]
d_tr <- build_data(prep, sp$train)
d_va <- build_data(prep, sp$val, d_tr$scale)
d_te <- build_data(prep, sp$test, d_tr$scale)
log_msg("数据 OK：train=%d val=%d test=%d", length(sp$train), length(sp$val), length(sp$test))

log_msg("== 3/5 torch 模型前向/反向（3 epochs） ==")
cfg <- CFG; cfg$epochs <- 3L; cfg$patience <- 3L; cfg$batch_size <- 512L
cfg$d_model <- 32L; cfg$n_layers <- 1L
device <- get_device()
for (mn in c("MLP_S", "MLP_L", "SeqOneTrans", "FTTransformer", "TCN", "TCTParallel")) {
  fit <- train_deep(mn, d_tr, d_va, prep, cfg, seed = 1L, device = device, verbose = FALSE)
  pred <- predict_deep(fit$model, d_te, prep, device)
  stopifnot(all(is.finite(pred)), all(pred > 0))
  log_msg("  %-14s OK", mn)
}

log_msg("== 4/5 E2 双优化器对照 ==")
fit_w <- train_deep("SeqOneTrans", d_tr, d_va, prep, cfg, seed = 2L,
                    optimizer = "adamw", device = device, verbose = FALSE)
fit_a <- train_deep("SeqOneTrans", d_tr, d_va, prep, cfg, seed = 2L,
                    optimizer = "adam", device = device, verbose = FALSE)
stopifnot(is.finite(fit_w$best_val), is.finite(fit_a$best_val))
log_msg("AdamW/Adam 双优化器 OK")

log_msg("== 5/5 审计与检测签名 ==")
wf <- tempfile(fileext = ".rds")
save_weights_rds(fit_w$model, wf)
aw <- audit_weights(wf)
stopifnot(nrow(aw$per_param) > 0)
qkv_df <- do.call(rbind, lapply(names(split_qkv(aw$state_dict)), function(nm) {
  m <- split_qkv(aw$state_dict)[[nm]]
  data.frame(param = nm, max_abs = max(abs(m)), l2_norm = sqrt(sum(m^2)),
             mean_abs = mean(abs(m)), n_elem = length(m))
}))
sig <- detect_signatures(aw$per_param, qkv_df)
stopifnot(is.logical(sig$params_dead))
bh <- audit_branch_health(
  { m <- build_model("TCTParallel", prep, cfg); m$to(device = device) },
  d_te, device)
stopifnot(is.finite(bh$gate))
log_msg("审计 OK：params_dead=%s，Q/K/V 拆分 %d 组", sig$params_dead, nrow(qkv_df))

# 消融开关
for (v in c("tcn", "attn", "avg")) {
  fv <- train_deep("TCTParallel", d_tr, d_va, prep, cfg, branch = v,
                   seed = 3L, device = device, verbose = FALSE)
  stopifnot(is.finite(fv$best_val))
}
log_msg("消融开关 OK")

log_msg("== 全部冒烟测试通过 ==")
