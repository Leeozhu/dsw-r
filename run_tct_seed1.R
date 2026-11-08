source("00_config.R"); source("01_prep_freMTPL2.R")
source("05_models_torch.R"); source("06_run_experiments.R")
init_dirs()

obj  <- readRDS("data/splits/fremtpl2.rds")
prep <- obj$prep; sp <- obj$splits[[1]]
d_tr <- build_data(prep, sp$train)
d_va <- build_data(prep, sp$val, d_tr$scale)
d_te <- build_data(prep, sp$test, d_tr$scale)

dev_cpu <- torch::torch_device("cpu")

fit <- train_deep("TCTParallel", d_tr, d_va, prep, CFG,
                  branch = "full", seed = CFG$seed_base + 1L,
                  device = dev_cpu, verbose = TRUE)

# 手动保存，predict 显式传 CPU
tag <- "fremtpl2_TCTParallel_split1_seed1"
pred <- predict_deep(fit$model, d_te, prep, device = dev_cpu)
m <- eval_task(prep$task, d_te$y, pred, d_te$exposure, 0.5)

saveRDS(list(dataset = "fremtpl2", model = "TCTParallel",
             split = 1L, seed = 1L,
             y = d_te$y, exposure = d_te$exposure, pred = pred,
             threshold = 0.5, branch = "full",
             device_note = "cpu (MPS divergent)"),
        file.path(CFG$paths$preds, paste0(tag, ".rds")))

save_weights_rds(fit$model, file.path(CFG$paths$weights, paste0(tag, ".rds")))

save_json(list(model = "TCTParallel", split = 1L, seed = 1L,
               optimizer = "adamw", lr = CFG$lr,
               weight_decay = CFG$weight_decay,
               epochs_run = fit$epochs_run, best_val = fit$best_val,
               threshold = 0.5, branch = "full",
               device_note = "cpu (MPS divergent)"),
          file.path(CFG$paths$weights, paste0(tag, "_meta.json")))

append_metrics(c(dataset = "fremtpl2", model = "TCTParallel",
                 split = 1L, seed = 1L, optimizer = "adamw",
                 epochs = fit$epochs_run, as.list(m)),
               file.path(CFG$paths$metrics, "metrics_long.csv"))

cat(sprintf("\nDONE: dev=%.4f gini=%.4f calib=%.4f epochs=%d best_val=%.5f\n",
            m["poisson_deviance"], m["gini_freq"], m["calib_rmse"],
            fit$epochs_run, fit$best_val))
