# =============================================================================
# 03_glm_baseline.R — 正确规格的 GLM 基准（每份划分各跑一次）
# 对应路线图：R1（factor 虚拟变量、ns() 样条、Exposure 只作 offset）
# 产出：每划分的测试集预测（rds）+ 指标（CSV）+ split1 的 bootstrap 指标 CI
# =============================================================================

source("00_config.R")
source("01_prep_freMTPL2.R")

# freMTPL2 频率 GLM：类别用 factor；连续用自然样条；raw/log 不重复入模。
# 关键：offset 必须写进公式（而非 glm() 的 offset 参数），否则 predict(newdata)
# 不会自动携带 offset，预测会退化为频率而非期望计数（审校 §二.1 修复）。
glm_formula_freq <- as.formula(
  ClaimNb ~ ns(BonusMalus, 4) + ns(DrivAge, 4) + ns(VehAge, 3) + ns(VehPower, 3) +
    Density_log + factor(Area) + factor(VehBrand) + factor(VehGas) + factor(Region) +
    offset(log(Exposure))
)

run_glm_fremtpl2 <- function(cfg = CFG) {
  suppressMessages(library(splines))
  obj <- readRDS(file.path(cfg$paths$splits, "fremtpl2.rds"))
  prep <- obj$prep; df <- prep$df

  for (sp in obj$splits) {
    s <- sp$split
    d_tr <- df[sp$train, ]; d_te <- df[sp$test, ]
    fit <- glm(glm_formula_freq, family = poisson(), data = d_tr)
    # offset 已在公式中，predict(newdata) 自动携带 log(Exposure) 并给出期望计数
    mu_hat  <- as.numeric(predict(fit, newdata = d_te, type = "response"))
    rate_hat <- mu_hat / d_te$Exposure

    m <- eval_task("poisson", d_te$ClaimNb, rate_hat, d_te$Exposure)
    saveRDS(list(dataset = "fremtpl2", model = "GLM", split = s, seed = 0L,
                 y = d_te$ClaimNb, exposure = d_te$Exposure, pred = rate_hat),
            file.path(cfg$paths$preds, sprintf("fremtpl2_GLM_split%d_seed0.rds", s)))
    append_metrics(c(dataset = "fremtpl2", model = "GLM", split = s, seed = 0L,
                     as.list(m)), file.path(cfg$paths$metrics, "metrics_long.csv"))
    log_msg("GLM split %d: PD=%.4f  Gini=%.4f", s, m["poisson_deviance"], m["gini_freq"])

    # split1 上做 bootstrap 指标 CI（替代"GLM 无法做 DM 检验"的缺口）
    if (s == 1L) {
      n <- nrow(d_te)
      ci_pd <- boot_ci(function() {
        b <- sample.int(n, replace = TRUE)
        poisson_deviance(d_te$ClaimNb[b], mu_hat[b])
      }, B = cfg$n_boot)
      ci_gini <- boot_ci(function() {
        b <- sample.int(n, replace = TRUE)
        gini_freq(d_te$ClaimNb[b], d_te$Exposure[b], rate_hat[b])
      }, B = cfg$n_boot)
      save_json(list(split = 1L, poisson_deviance = ci_pd, gini_freq = ci_gini),
                file.path(cfg$paths$metrics, "glm_bootstrap_ci.json"))
    }
  }
  invisible(TRUE)
}

if (sys.nframe() == 0L) { init_dirs(); run_glm_fremtpl2() }
