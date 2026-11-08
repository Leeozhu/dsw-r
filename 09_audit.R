# =============================================================================
# 09_audit.R — 模型内部健康审计与失效检测清单（RQ2/RQ3 核心产出）
# 含注意力熵审计（检出"结构恒等化"：seq_len=1 时熵恒为 0）；检测矩阵增注意力熵列
# =============================================================================

source("00_config.R")
source("01_prep_freMTPL2.R")
source("05_models_torch.R")

FLOAT32_MIN_NORMAL <- 1.1754944e-38

# ---- 权重范数审计 -------------------------------------------------------------
audit_weights <- function(weights_rds_path) {
  sd <- readRDS(weights_rds_path)
  rows <- lapply(names(sd), function(nm) {
    w <- sd[[nm]]
    data.frame(param = nm, l2_norm = sqrt(sum(w^2)), max_abs = max(abs(w)),
               mean_abs = mean(abs(w)), n_elem = length(w), stringsAsFactors = FALSE)
  })
  df <- do.call(rbind, rows)
  # 精确匹配三种前缀，避免 attn_blocks.1. 被 blocks.1. 先行破坏
  df$group <- gsub("^(attn_blocks|tcn_blocks|blocks)\\.\\d+\\.", "", df$param)
  df$group <- gsub("^tokenizer\\.", "", df$group)
  list(per_param = df, state_dict = sd)
}

split_qkv <- function(sd) {
  out <- list()
  for (nm in names(sd)) {
    if (grepl("in_proj_weight$", nm)) {
      w <- sd[[nm]]; d <- nrow(w) / 3
      out[[paste0(nm, ".W_Q")]] <- w[seq_len(d), , drop = FALSE]
      out[[paste0(nm, ".W_K")]] <- w[d + seq_len(d), , drop = FALSE]
      out[[paste0(nm, ".W_V")]] <- w[2 * d + seq_len(d), , drop = FALSE]
    }
  }
  out
}

# ---- 分支/组件激活审计（TCT 专用） --------------------------------------------
audit_branch_health <- function(model, d_te, device) {
  t <- to_tensors(d_te, device)
  model$eval()
  br <- with_no_grad({
    o <- model$branch_outputs(t$x_num, t$x_cat)
    list(tcn = as_array(o$tcn$cpu()), attn = as_array(o$attn$cpu()),
         gate = as_array(torch_sigmoid(model$gate_logit)$cpu()))
  })
  list(gate = br$gate,
       branch_activation = data.frame(
         branch = c("tcn", "attn"),
         mean_abs = c(mean(abs(br$tcn)), mean(abs(br$attn))),
         sd_abs   = c(sd(abs(br$tcn)),  sd(abs(br$attn))),
         frac_near_zero = c(mean(abs(br$tcn) < 1e-6), mean(abs(br$attn) < 1e-6))))
}

# ---- 注意力熵审计：检出"结构恒等化"（seq_len=1 时熵恒为 0） -------------------
# mirror 各模型 forward，捕获 nn_multihead_attention 的第二返回值（注意力权重）。
# 返回每层注意力熵（batch x query token 的均值）。
compute_attention_entropy <- function(model, model_name, d_te, device) {
  t <- to_tensors(d_te, device)
  model$eval()

  extract_entropy <- function(attn_out) {
    if (!is.list(attn_out) || length(attn_out) < 2) return(NULL)
    aw <- attn_out[[2]]                         # (B, L, L)
    if (is.null(aw)) return(NULL)
    p <- aw$clamp(min = 1e-12)
    ent <- -torch_sum(p * torch_log(p), dim = 3)   # (B, L)
    as_array(ent$cpu())
  }

  step_block <- function(h, blk, li, out) {
    hn <- blk$ln1(h)
    a <- blk$attn(hn, hn, hn)
    ent <- extract_entropy(a); if (!is.null(ent)) out[[li]] <<- ent
    a_out <- if (is.list(a)) a[[1]] else a
    h <- h + blk$drop(a_out)
    h + blk$drop(blk$ffn(blk$ln2(h)))
  }

  res <- with_no_grad({
    out <- list()
    if (model_name == "SeqOneTransformer") {
      embs <- lapply(seq_along(model$cat_embs), function(j) model$cat_embs[[j]](t$x_cat[, j]))
      z <- torch_cat(c(list(t$x_num), embs), dim = 2)
      h <- model$input_proj(z)$unsqueeze(2)
      for (li in seq_along(model$blocks)) h <- step_block(h, model$blocks[[li]], li, out)
    } else if (model_name == "FTTransformer") {
      h <- model$tokenizer(t$x_num, t$x_cat)
      for (li in seq_along(model$blocks)) h <- step_block(h, model$blocks[[li]], li, out)
    } else if (model_name == "TCTParallel") {
      ha <- model$tokenizer(t$x_num, t$x_cat)
      for (li in seq_along(model$attn_blocks)) ha <- step_block(ha, model$attn_blocks[[li]], li, out)
    }
    out
  })

  if (!length(res)) return(NULL)
  data.frame(
    layer        = seq_along(res),
    mean_entropy = sapply(res, mean),
    sd_entropy   = sapply(res, function(x) if (length(x) > 1) sd(as.vector(x)) else NA_real_),
    min_entropy  = sapply(res, function(x) min(as.vector(x))))
}

# ---- 失效检测判定：把审计证据翻译成布尔签名 ------------------------------------
detect_signatures <- function(per_param, qkv_df, attn_entropy_df = NULL) {
  sig <- list()
  attn_rows <- per_param[grepl("attn|in_proj|out_proj", per_param$param), ]
  sig$params_dead <- nrow(attn_rows) > 0 && any(attn_rows$max_abs < 1e-30)
  if (!is.null(qkv_df) && nrow(qkv_df)) {
    qk <- qkv_df[grepl("\\.W_[QK]", qkv_df$param), ]
    vv <- qkv_df[grepl("\\.W_V", qkv_df$param), ]
    sig$qk_selective_death <- nrow(qk) > 0 &&
      all(qk$max_abs < 1e-30) && all(vv$max_abs > 1e-10)
  } else sig$qk_selective_death <- NA
  # 签名 3：注意力熵坍塌（参数健康但注意力恒等化）
  if (!is.null(attn_entropy_df) && nrow(attn_entropy_df)) {
    sig$attention_entropy_collapse <- all(attn_entropy_df$mean_entropy < 1e-6)
    sig$min_attention_entropy <- min(attn_entropy_df$mean_entropy)
  } else {
    sig$attention_entropy_collapse <- NA
    sig$min_attention_entropy <- NA
  }
  sig
}

# ---- 主流程 -------------------------------------------------------------------
run_audit <- function(dataset = "fremtpl2", model_name = "TCTParallel",
                      split = 1L, seed = 1L, cfg = CFG) {
  device <- get_device()
  tag <- sprintf("%s_%s_split%d_seed%d", dataset, model_name, split, seed)
  wf <- file.path(cfg$paths$weights, paste0(tag, ".rds"))
  if (!file.exists(wf)) { log_msg("跳过（无权重）：%s", tag); return(invisible(NULL)) }
  meta_f <- file.path(cfg$paths$weights, paste0(tag, "_meta.json"))
  meta <- if (file.exists(meta_f)) jsonlite::read_json(meta_f) else list()

  aw <- audit_weights(wf)
  qkv <- split_qkv(aw$state_dict)
  qkv_df <- if (length(qkv))
    do.call(rbind, lapply(names(qkv), function(nm)
      data.frame(param = nm, l2_norm = sqrt(sum(qkv[[nm]]^2)),
                 max_abs = max(abs(qkv[[nm]])), mean_abs = mean(abs(qkv[[nm]])),
                 n_elem = length(qkv[[nm]])))) else data.frame()

  # 实例化模型并载权重（用于注意力熵与分支激活）
  obj <- readRDS(file.path(cfg$paths$splits, paste0(dataset, ".rds")))
  prep <- obj$prep; sp <- obj$splits[[split]]
  builder <- if (prep$task == "binary") build_data_bin else build_data
  d_tr <- builder(prep, sp$train); d_te <- builder(prep, sp$test, d_tr$scale)

  branch <- if (!is.null(meta$branch)) meta$branch else "full"
  model <- build_model(model_name, prep, cfg, branch)$to(device = device)
  model$load_state_dict(lapply(readRDS(wf),
    function(a) torch_tensor(a, dtype = torch_float(), device = device)))

  attn_ent <- tryCatch(compute_attention_entropy(model, model_name, d_te, device),
                       error = function(e) {
                         log_msg("注意力熵计算失败：%s", e$message); NULL })

  sig <- detect_signatures(aw$per_param, qkv_df, attn_ent)
  audit <- c(list(tag = tag, optimizer_meta = meta,
                  n_collapsed = sum(aw$per_param$max_abs < 1e-30)),
             sig)

  # TCT 分支健康
  if (grepl("TCTParallel", model_name)) {
    bh <- audit_branch_health(model, d_te, device)
    audit$gate_value <- bh$gate
    audit$branch_activation_summary <- as.list(setNames(
      round(bh$branch_activation$mean_abs, 8), bh$branch_activation$branch))
    audit$branch_dead <- any(bh$branch_activation$mean_abs < 1e-6)
    write.csv(bh$branch_activation,
              file.path(cfg$paths$metrics, paste0(tag, "_branch_activation.csv")),
              row.names = FALSE)
  }

  if (!is.null(attn_ent))
    write.csv(attn_ent,
              file.path(cfg$paths$metrics, paste0(tag, "_attn_entropy.csv")),
              row.names = FALSE)

  write.csv(aw$per_param,
            file.path(cfg$paths$metrics, paste0(tag, "_weight_audit.csv")),
            row.names = FALSE)
  if (nrow(qkv_df)) write.csv(qkv_df,
    file.path(cfg$paths$metrics, paste0(tag, "_qkv_audit.csv")), row.names = FALSE)
  save_json(audit, file.path(cfg$paths$metrics, paste0(tag, "_audit.json")))
  log_msg("审计完成 %s：collapsed=%d, params_dead=%s, qk_selective=%s, attn_entropy_collapse=%s",
          tag, audit$n_collapsed, sig$params_dead,
          as.character(sig$qk_selective_death),
          as.character(sig$attention_entropy_collapse))
  invisible(audit)
}

# ---- 批量审计：E2 全部细胞 -> 检测矩阵 ------------------------------------------
run_detection_matrix <- function(cfg = CFG) {
  cells <- list(
    A = list(model = "FTTransformer", optimizer = "adamw"),
    B = list(model = "FTTransformer", optimizer = "adam"),
    C = list(model = "SeqOneTrans",   optimizer = "adamw"),
    D = list(model = "SeqOneTrans",   optimizer = "adam")
  )
  rows <- list()
  for (cid in names(cells)) {
    cell <- cells[[cid]]
    run_model <- sprintf("E2%s_%s_%s", cid, cell$model, cell$optimizer)
    for (s in seq_len(cfg$e2_splits)) for (k in seq_len(cfg$e2_seeds)) {
      a <- run_audit("fremtpl2", run_model, s, k, cfg)
      if (!is.null(a))
        rows[[length(rows) + 1]] <- data.frame(
          cell = cid, model = cell$model, optimizer = cell$optimizer,
          split = s, seed = k,
          params_dead = a$params_dead,
          qk_selective_death = as.character(a$qk_selective_death),
          attention_entropy_collapse = as.character(a$attention_entropy_collapse),
          min_attention_entropy = a$min_attention_entropy,
          n_collapsed = a$n_collapsed)
    }
  }
  mat <- do.call(rbind, rows)
  write.csv(mat, file.path(cfg$paths$metrics, "e2_detection_matrix.csv"),
            row.names = FALSE)
  log_msg("E2 检测矩阵已写盘：%d 行", nrow(mat))
  mat
}

if (sys.nframe() == 0L) { init_dirs(); run_audit(); run_detection_matrix() }
