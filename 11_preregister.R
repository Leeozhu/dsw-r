# =============================================================================
# 11_preregister.R — E2 实验判定标准的预注册（研究方案 §6.7）
# 在任何 E2 实验运行【之前】执行一次：生成带时间戳的判定标准 JSON。
# 该文件一旦生成不得修改；重复运行只会报告已存在的文件及其哈希。
# 用法：Rscript 11_preregister.R
# =============================================================================

source("00_config.R")

PREREG_TEXT <- list(
  title = "E2 受控失效注入：2x2 因子设计的判定标准预注册",
  design = "输入表示 {FT tokenizer 正确, seq_len=1 误用} x 优化器 {AdamW 健康, Adam 耦合 L2 病理}，细胞 A/B/C/D",
  verdict_criteria = list(
    reproduced = list(
      definition = "Q/K 投影 max_abs < 1e-30 且 V/O 健康（max_abs > 1e-3），即 qk_selective_death 签名为真",
      reporting  = "正文报告坍塌轨迹与分组范数表；细胞 D 标记为'复现'"
    ),
    partial = list(
      definition = "Q/K 范数显著低于健康基线（< 基线 1/100）但未触及 1e-30",
      reporting  = "正文报告'实现依赖性'发现：坍塌幅度对 float32 精度/eps/步数敏感"
    ),
    not_reproduced = list(
      definition = "Q/K 与健康基线无数量级差异",
      reporting  = "正文作为独立发现报告：旧论文坍塌数字对实现细节敏感；强化'此类数字必须溯源审计工件'的治理结论；旧 .pt 检查点解剖（第 0 阶段）的解释权重上升"
    )
  ),
  commitments = c(
    "无论三档结果如何，表 3（E2 检测矩阵）如实填写全部四个细胞",
    "不以'结果不理想'为由省略任何细胞",
    "E2 全部中间工件（权重 rds、meta json）随论文开源",
    "E1 判读预案见研究方案 §6.2，与本文件共同构成结果-叙事预注册"
  )
)

preregister_e2 <- function(cfg = CFG) {
  f <- file.path(cfg$paths$metrics, "preregistration_e2.json")
  if (file.exists(f)) {
    log_msg("预注册文件已存在（不得修改）：%s", f)
    log_msg("如需核验完整性，请比对首次生成时记录的哈希。")
    return(invisible(readChar(f, file.size(f))))
  }
  payload <- c(PREREG_TEXT,
               list(created_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"),
                    r_version = R.version.string,
                    note = "本文件由 11_preregister.R 首次运行生成，此后不得修改"))
  save_json(payload, f)
  log_msg("预注册完成：%s（创建时间已固化）", f)
  log_msg("请将本文件连同时间戳存档（如提交至 OSF 或附入论文补充材料）。")
  invisible(f)
}

if (sys.nframe() == 0L) { init_dirs(); preregister_e2() }
