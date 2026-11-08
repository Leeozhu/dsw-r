# =============================================================================
# run_all.R — 一键全流程（研究方案 v1.2：透明度光谱与模型可验证性）
# 用法：
#   Rscript run_all.R           # freMTPL2 全流程（E1+E2+TCT 消融）
#   Rscript run_all.R porto     # E4 Porto 案例
#   Rscript run_all.R all       # 全部
#   Rscript run_all.R quick     # 快速模式（2 划分 x 1 种子，验证管线）
# =============================================================================

args <- commandArgs(trailingOnly = TRUE)
mode <- if (length(args)) args[1] else "fremtpl2"

# 先 source 配置（定义 CFG / init_dirs / log_msg），再初始化目录
source("00_config.R")
init_dirs()

if (mode == "quick") {
  CFG$n_splits <- 2L; CFG$n_seeds <- 1L; CFG$e2_splits <- 1L; CFG$e2_seeds <- 1L
  CFG$epochs <- 30L; CFG$patience <- 8L
  log_msg("快速模式")
}

source("11_preregister.R")   # 预注册必须先于任何 E2 实验（§6.7）
preregister_e2()

source("01_prep_freMTPL2.R")
source("02_prep_porto.R")
source("03_glm_baseline.R")
source("04_gbm_models.R")
source("04a_gam_models.R")
source("05_models_torch.R")
source("06_run_experiments.R")
source("07_evaluation.R")
source("08_interpret.R")
source("09_audit.R")
source("10_report.R")

run_fre <- mode %in% c("fremtpl2", "all", "quick")
run_por <- mode %in% c("porto", "all")

if (run_fre) {
  prep_fremtpl2_all()
  run_glm_fremtpl2()          # L0
  run_gam_fremtpl2()          # L1
  run_xgb_dataset("fremtpl2") # L2
  run_e1_spectrum()           # L3/L4
  run_e2_failure_injection()  # E2 失效注入
  run_ablation_fremtpl2()     # TCT 严格消融
}
if (run_por) {
  prep_porto_all()
  run_xgb_dataset("porto")
  run_e4_porto()
}

collect_metrics(); summarize_metrics(); run_pairwise_tests()

if (run_fre) {
  tryCatch(run_interpret("TCTParallel"),
           error = function(e) log_msg("interpret 跳过：%s", e$message))
  tryCatch(run_detection_matrix(),
           error = function(e) log_msg("检测矩阵跳过：%s", e$message))
  for (mn in c("TCTParallel", "FTTransformer"))
    tryCatch(run_audit("fremtpl2", mn),
             error = function(e) log_msg("audit %s 跳过：%s", mn, e$message))
}

build_paper_tables()
log_msg("== 全流程完成。结果见 outputs/ ==")
