# =============================================================================
# 05_models_torch.R — 深度模型（透明度光谱 L3/L4）与通用训练器
# 模型清单：
#   MLP_S / MLP_L     —— L3/L4：小型与大型 MLP（复杂度梯度）
#   FTTransformer     —— L4：feature tokenizer 正确用法（每特征一个 token）
#   SeqOneTransformer —— E2 专用：seq_len=1 伪序列【误用对照】
#   TCTParallel       —— L4：旧架构延续（TCN 并行 Transformer 门控融合），
#                        branch 开关支持严格删减式消融
# 训练器支持 optimizer = "adamw"（主实验）/ "adam"（耦合 L2，仅失效注入用）
# =============================================================================

source("00_config.R")
library(torch)

# ---- 特征分词器：数值 -> 每特征一个 token；类别 -> embedding ------------------
FeatureTokenizer <- nn_module(
  "FeatureTokenizer",
  initialize = function(n_num, cat_levels, d_model) {
    self$n_num <- n_num; self$cat_levels <- cat_levels; self$d_model <- d_model
    if (n_num > 0) {
      self$num_weight <- nn_parameter(torch_randn(n_num, d_model) * 0.02)
      self$num_bias   <- nn_parameter(torch_zeros(n_num, d_model))
    }
    self$cat_embs <- nn_module_list(lapply(cat_levels, function(L) nn_embedding(L, d_model)))
    self$cls_token <- nn_parameter(torch_randn(1, 1, d_model) * 0.02)
  },
  forward = function(x_num, x_cat) {
    B <- if (!is.null(x_num)) x_num$size(1) else x_cat$size(1)
    toks <- list()
    if (self$n_num > 0)
      toks <- c(toks, list(x_num$unsqueeze(3) * self$num_weight + self$num_bias))
    if (length(self$cat_levels) > 0)
      toks <- c(toks, list(torch_stack(lapply(seq_along(self$cat_levels),
                           function(j) self$cat_embs[[j]](x_cat[, j])), dim = 2)))
    tok <- torch_cat(toks, dim = 2)
    cls <- self$cls_token$expand(c(B, 1L, self$d_model))
    torch_cat(list(cls, tok), dim = 2)                      # (B, 1+L, d)，CLS 首位
  }
)

# ---- pre-LN Transformer 编码器块（显式实现，便于组件级审计） ------------------
EncoderBlock <- nn_module(
  "EncoderBlock",
  initialize = function(d_model, n_heads, dropout) {
    self$ln1 <- nn_layer_norm(d_model)
    self$attn <- nn_multihead_attention(d_model, n_heads, dropout = dropout,
                                        batch_first = TRUE)
    self$ln2 <- nn_layer_norm(d_model)
    self$ffn <- nn_sequential(nn_linear(d_model, 4 * d_model), nn_gelu(),
                              nn_dropout(dropout), nn_linear(4 * d_model, d_model))
    self$drop <- nn_dropout(dropout)
  },
  forward = function(x) {
    h <- self$ln1(x)
    a <- self$attn(h, h, h)
    if (is.list(a)) a <- a[[1]]
    x <- x + self$drop(a)
    x + self$drop(self$ffn(self$ln2(x)))
  }
)

# ---- MLP（hidden 决定 L3/L4 分档） ---------------------------------------------
MLPRegressor <- nn_module(
  "MLPRegressor",
  initialize = function(n_num, cat_levels, hidden = c(64L), emb_dim = 8L,
                        dropout = 0.1) {
    self$cat_embs <- nn_module_list(lapply(cat_levels, function(L) nn_embedding(L, emb_dim)))
    in_dim <- n_num + emb_dim * length(cat_levels)
    layers <- list(); d <- in_dim
    for (h in hidden) {
      layers <- c(layers, list(nn_linear(d, h), nn_relu(), nn_dropout(dropout)))
      d <- h
    }
    self$body <- do.call(nn_sequential, layers)   # 避免依赖 rlang 的 !!! 准引用
    self$head <- nn_linear(d, 1L)
  },
  forward = function(x_num, x_cat) {
    embs <- lapply(seq_along(self$cat_embs), function(j) self$cat_embs[[j]](x_cat[, j]))
    z <- torch_cat(c(list(x_num), embs), dim = 2)
    self$head(self$body(z))$squeeze(2)
  }
)

# ---- E2 误用对照：整行 -> (B,1,F) 单 token（注意力恒等化） ----------------------
SeqOneTransformer <- nn_module(
  "SeqOneTransformer",
  initialize = function(n_num, cat_levels, d_model = 64L, n_heads = 4L,
                        n_layers = 2L, dropout = 0.1, emb_dim = 8L) {
    self$cat_embs <- nn_module_list(lapply(cat_levels, function(L) nn_embedding(L, emb_dim)))
    in_dim <- n_num + emb_dim * length(cat_levels)
    self$input_proj <- nn_linear(in_dim, d_model)
    self$blocks <- nn_module_list(lapply(seq_len(n_layers),
                                         function(i) EncoderBlock(d_model, n_heads, dropout)))
    self$head <- nn_linear(d_model, 1L)
  },
  forward = function(x_num, x_cat) {
    embs <- lapply(seq_along(self$cat_embs), function(j) self$cat_embs[[j]](x_cat[, j]))
    z <- torch_cat(c(list(x_num), embs), dim = 2)
    h <- self$input_proj(z)$unsqueeze(2)                    # (B, 1, d) 单 token
    for (i in seq_len(length(self$blocks))) h <- self$blocks[[i]](h)
    self$head(h$squeeze(2))$squeeze(2)
  }
)

# ---- FT-Transformer（L4，正确用法） --------------------------------------------
FTTransformer <- nn_module(
  "FTTransformer",
  initialize = function(n_num, cat_levels, d_model = 64L, n_heads = 4L,
                        n_layers = 2L, dropout = 0.1) {
    self$tokenizer <- FeatureTokenizer(n_num, cat_levels, d_model)
    self$blocks <- nn_module_list(lapply(seq_len(n_layers),
                                         function(i) EncoderBlock(d_model, n_heads, dropout)))
    self$head <- nn_linear(d_model, 1L)
  },
  forward = function(x_num, x_cat) {
    h <- self$tokenizer(x_num, x_cat)
    for (i in seq_len(length(self$blocks))) h <- self$blocks[[i]](h)
    self$head(h[, 1, ])$squeeze(2)
  }
)

# ---- 因果扩张卷积（供 TCN / TCT 使用） ------------------------------------------
CausalConv1d <- nn_module(
  "CausalConv1d",
  initialize = function(ch, kernel = 3L, dilation = 1L, dropout = 0.1) {
    self$pad <- as.integer((kernel - 1) * dilation)
    self$conv <- nn_conv1d(ch, ch, kernel_size = kernel,
                           padding = self$pad, dilation = dilation)
    self$norm <- nn_layer_norm(ch)
    self$drop <- nn_dropout(dropout)
  },
  forward = function(x) {                                   # (B, C, L)
    L <- x$size(3)
    y <- self$conv(x)
    if (self$pad > 0) y <- torch_narrow(y, dim = 3, start = 1, length = L)
    y <- nnf_relu(y)
    y <- self$norm(y$permute(c(1, 3, 2)))$permute(c(1, 3, 2))
    x + self$drop(y)
  }
)

TCNNet <- nn_module(
  "TCNNet",
  initialize = function(n_num, cat_levels, d_model = 64L, kernel = 3L,
                        dilations = c(1L, 2L, 4L), dropout = 0.1) {
    self$tokenizer <- FeatureTokenizer(n_num, cat_levels, d_model)
    self$blocks <- nn_module_list(lapply(dilations,
                                         function(dl) CausalConv1d(d_model, kernel, dl, dropout)))
    self$head <- nn_linear(d_model, 1L)
  },
  forward = function(x_num, x_cat) {
    h <- self$tokenizer(x_num, x_cat)$permute(c(1, 3, 2))
    for (i in seq_len(length(self$blocks))) h <- self$blocks[[i]](h)
    self$head(h$mean(dim = 3))$squeeze(2)
  }
)

# ---- TCT-Parallel（L4 延续旧架构）；branch 开关 = 严格删减式消融 ------------------
TCTParallel <- nn_module(
  "TCTParallel",
  initialize = function(n_num, cat_levels, d_model = 64L, n_heads = 4L,
                        n_layers = 2L, kernel = 3L, dilations = c(1L, 2L, 4L),
                        dropout = 0.1, branch = c("full", "tcn", "attn", "avg")) {
    self$branch <- match.arg(branch)
    self$tokenizer <- FeatureTokenizer(n_num, cat_levels, d_model)
    self$attn_blocks <- nn_module_list(lapply(seq_len(n_layers),
                                              function(i) EncoderBlock(d_model, n_heads, dropout)))
    self$tcn_blocks <- nn_module_list(lapply(dilations,
                                             function(dl) CausalConv1d(d_model, kernel, dl, dropout)))
    self$gate_logit <- nn_parameter(torch_zeros(1))
    self$head <- nn_linear(d_model, 1L)
  },
  branch_outputs = function(x_num, x_cat) {                 # 审计接口：分支表示
    tok <- self$tokenizer(x_num, x_cat)
    ha <- tok
    for (i in seq_len(length(self$attn_blocks))) ha <- self$attn_blocks[[i]](ha)
    ht <- tok$permute(c(1, 3, 2))
    for (i in seq_len(length(self$tcn_blocks))) ht <- self$tcn_blocks[[i]](ht)
    list(tcn = ht$mean(dim = 3), attn = ha[, 1, ])
  },
  forward = function(x_num, x_cat) {
    br <- self$branch_outputs(x_num, x_cat)
    g <- torch_sigmoid(self$gate_logit)
    fused <- switch(self$branch,
                    full = g * br$tcn + (1 - g) * br$attn,
                    avg  = 0.5 * br$tcn + 0.5 * br$attn,
                    tcn  = br$tcn,
                    attn = br$attn)
    self$head(fused)$squeeze(2)
  }
)

# 模型注册表：MLP_S/MLP_L 只是 hidden 不同（透明度光谱的复杂度梯度）
MODEL_REGISTRY <- list(
  MLP_S = function(n_num, cat_levels, dropout = 0.1, ...)
    MLPRegressor(n_num, cat_levels, hidden = c(64L), dropout = dropout),
  MLP_L = function(n_num, cat_levels, dropout = 0.1, ...)
    MLPRegressor(n_num, cat_levels, hidden = c(256L, 128L, 64L), dropout = dropout),
  SeqOneTrans   = SeqOneTransformer,
  FTTransformer = FTTransformer,
  TCN           = TCNNet,
  TCTParallel   = TCTParallel
)

build_model <- function(model_name, prep, cfg = CFG, branch = "full") {
  base <- list(n_num = length(prep$num_features), cat_levels = prep$cat_levels,
               dropout = cfg$dropout)
  enc  <- c(base, list(d_model = cfg$d_model))
  full <- c(enc,  list(n_heads = cfg$n_heads, n_layers = cfg$n_layers))
  args <- switch(model_name,
                 MLP_S = base, MLP_L = base,
                 TCN   = enc,
                 SeqOneTrans = full, FTTransformer = full,
                 TCTParallel = c(full, list(branch = branch)),
                 stop("未知模型：", model_name))
  do.call(MODEL_REGISTRY[[model_name]], args)
}

# ---------------------------------------------------------------------------
# 通用训练器
#   optimizer = "adamw"：解耦衰减（主实验）
#   optimizer = "adam" ：耦合 L2（仅 E2 失效注入复现"静默坍塌"病理）
# ---------------------------------------------------------------------------
to_tensors <- function(d, device) {
  list(x_num = torch_tensor(d$x_num, dtype = torch_float(), device = device),
       x_cat = torch_tensor(d$x_cat, dtype = torch_long(),  device = device),
       y     = torch_tensor(d$y,     dtype = torch_float(), device = device),
       exposure = if (!is.null(d$exposure))
         torch_tensor(d$exposure, dtype = torch_float(), device = device) else NULL)
}

compute_loss <- function(out, t, task, pos_weight_t = NULL) {
  if (task == "poisson") {
    log_mu <- out + torch_log(t$exposure$clamp(min = 1e-10))
    torch_mean(torch_exp(log_mu) - t$y * log_mu)
  } else if (task == "binary") {
    nnf_binary_cross_entropy_with_logits(out, t$y, pos_weight = pos_weight_t)
  } else {
    nnf_mse_loss(out, torch_log1p(t$y))
  }
}

val_metric <- function(out, d, task) {
  p <- as_array(out$cpu())
  if (task == "poisson") return(poisson_deviance(d$y, exp(p) * d$exposure))
  if (task == "binary")  return(-auc_binary(d$y, 1 / (1 + exp(-p))))
  rmse_score(log1p(d$y), p)
}

train_deep <- function(model_name, d_train, d_val, prep, cfg = CFG,
                       branch = "full", pos_weight = 1, seed = 42L,
                       optimizer = c("adamw", "adam"),
                       device = get_device(), verbose = TRUE) {
  optimizer <- match.arg(optimizer)
  set_all_seeds(seed)
  model <- build_model(model_name, prep, cfg, branch)$to(device = device)

  t_tr <- to_tensors(d_train, device); t_va <- to_tensors(d_val, device)
  pos_w_t <- if (prep$task == "binary")
    torch_tensor(pos_weight, dtype = torch_float(), device = device) else NULL

  opt <- if (optimizer == "adamw")
    optim_adamw(model$parameters, lr = cfg$lr, weight_decay = cfg$weight_decay)
  else
    optim_adam(model$parameters, lr = cfg$lr, weight_decay = cfg$weight_decay)  # 耦合 L2

  n <- length(d_train$y)
  best <- Inf; best_state <- NULL; wait <- 0L; epochs_run <- 0L

  for (epoch in seq_len(cfg$epochs)) {
    model$train()
    ord <- sample.int(n)
    for (b in split(ord, ceiling(ord / cfg$batch_size))) {
      idx <- torch_tensor(b, dtype = torch_long(), device = device)
      bt <- list(x_num = t_tr$x_num$index_select(1, idx),
                 x_cat = t_tr$x_cat$index_select(1, idx),
                 y = t_tr$y$index_select(1, idx),
                 exposure = if (!is.null(t_tr$exposure))
                   t_tr$exposure$index_select(1, idx) else NULL)
      opt$zero_grad()
      loss <- compute_loss(model(bt$x_num, bt$x_cat), bt, prep$task, pos_w_t)
      loss$backward()
      opt$step()
    }
    epochs_run <- epoch
    model$eval()
    vm <- with_no_grad({ val_metric(model(t_va$x_num, t_va$x_cat), d_val, prep$task) })
    if (vm < best - 1e-5) {
      best <- vm; wait <- 0L
      best_state <- lapply(model$state_dict(), function(x) as_array(x$cpu()))
    } else {
      wait <- wait + 1L
      if (wait >= cfg$patience) break
    }
  }
  if (!is.null(best_state))
    model$load_state_dict(lapply(best_state,
                                 function(a) torch_tensor(a, dtype = torch_float(),
                                                          device = device)))
  if (verbose) log_msg("  %s(branch=%s, %s)：epochs=%d  best_val=%.5f",
                       model_name, branch, optimizer, epochs_run, best)
  list(model = model, best_val = best, epochs_run = epochs_run, optimizer = optimizer)
}

predict_deep <- function(model, d_test, prep, device = get_device()) {
  t <- to_tensors(d_test, device)
  model$eval()
  out <- with_no_grad({ as_array(model(t$x_num, t$x_cat)$cpu()) })
  if (prep$task == "poisson") return(exp(out))
  if (prep$task == "binary")  return(1 / (1 + exp(-out)))
  pmax(expm1(out), 0)
}

save_weights_rds <- function(model, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  saveRDS(lapply(model$state_dict(), function(x) as_array(x$cpu())), path)
  invisible(path)
}
