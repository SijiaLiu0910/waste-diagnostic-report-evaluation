# Cumulative link mixed models for reliability and logical coherence.
# Families: six overall LRTs and eight prespecified contrasts, each Holm-adjusted.
# =============================================================================
# clmm_models.R
#   score ~ perception * rag + rater +
#           (1|sampling_point) + (1|observation_block) + (1|report_id)
# =============================================================================

.script <- local({a <- commandArgs(trailingOnly = FALSE)
                  f <- sub("^--file=", "", a[grepl("^--file=", a)])
                  if (length(f)) f[1] else ""})
ROOT <- normalizePath(file.path(if (nzchar(.script)) dirname(.script) else ".",
                                "..", ".."), mustWork = FALSE)
LONG_CSV <- file.path(ROOT, "results/evaluation/report_quality/ratings_long.csv")
OUT_DIR  <- file.path(ROOT, "results/recomputed/statistics")
OUT_MODEL  <- file.path(OUT_DIR, "recommendation_model_coefficients.csv")
OUT_PAIR   <- file.path(OUT_DIR, "recommendation_pairwise_contrasts.csv")
OUT_DIAG   <- file.path(OUT_DIR, "model_diagnostics.md")

dir.create(OUT_DIR, recursive=TRUE, showWarnings=FALSE)
BOOT_SEED <- 20261122
Z95 <- qnorm(0.975)

suppressMessages({
  library(ordinal)
  library(emmeans)
})

d <- read.csv(LONG_CSV, fileEncoding = "UTF-8-BOM", check.names = FALSE,
              stringsAsFactors = FALSE)
stopifnot(nrow(d) == 240)
d$perception <- factor(d$perception_pathway, levels = c("MLLM", "YOLO"))
d$rag        <- factor(d$rag_condition, levels = c("without_RAG", "with_RAG"))
d$rater      <- factor(d$rater, levels = c("Rater1", "Rater2"))

RF <- "(1|sampling_site_id) + (1|observation_block_id) + (1|report_id)"
score_map <- c(reliability = "reliability", logical_coherence = "logical_coherence")

# Quote text fields so commas in English explanations remain within one cell.
write_parted_csv <- function(path, parts) {
  con <- file(path, open="wb")
  on.exit(close(con))
  writeBin(as.raw(c(0xef, 0xbb, 0xbf)), con)
  for (df in parts) {
    write.table(df, con, sep=",", row.names=FALSE, na="NA", qmethod="double")
    writeLines("", con)
  }
}

fmt <- function(x, digits = 4) ifelse(is.na(x), "NA", formatC(x, digits = digits, format = "f"))

coef_rows   <- list()
lrt_rows    <- list()
thresh_rows <- list()
var_rows    <- list()
conv_rows   <- list()
pair_rows   <- list()

diag_notes  <- list()
aux_tests   <- list()

for (sc in names(score_map)) {
  dd <- d
  dd$score <- factor(dd[[sc]], ordered = TRUE)

  m_full <- clmm(as.formula(paste("score ~ perception*rag + rater +", RF)), data = dd)
  m_add  <- clmm(as.formula(paste("score ~ perception + rag + rater +", RF)), data = dd)
  m_np   <- clmm(as.formula(paste("score ~ rag + rater +", RF)), data = dd)
  m_nr   <- clmm(as.formula(paste("score ~ perception + rater +", RF)), data = dd)

  cf <- summary(m_full)$coefficients
  fe_names <- c("perceptionYOLO" = "Perception pathway: YOLO versus MLLM",
                "ragwith_RAG" = "Knowledge condition: with versus without RAG",
                "raterRater2" = "Rater: Rater2 versus Rater1",
                "perceptionYOLO:ragwith_RAG" = "Interaction: YOLO × with RAG")
  for (fn in names(fe_names)) {
    beta <- cf[fn, "Estimate"]; se <- cf[fn, "Std. Error"]
    z <- cf[fn, "z value"]; p_wald <- cf[fn, "Pr(>|z|)"]
    coef_rows[[length(coef_rows) + 1]] <- data.frame(
      score_dimension = sc, model = "Interaction CLMM (cumulative logit)", effect = fe_names[fn],
      effect_code = fn,
      beta = round(beta, 4), SE = round(se, 4), z = round(z, 3),
      OR = round(exp(beta), 4),
      OR_ci95_lower = round(exp(beta - Z95 * se), 4),
      OR_ci95_upper = round(exp(beta + Z95 * se), 4),
      p_wald_raw = signif(p_wald, 4),
      p_holm = "NA (fixed-effect Wald tests are outside the two Holm families; see LRT results)",
      stringsAsFactors = FALSE
    )
  }

  lrt_inter <- anova(m_add, m_full)
  lrt_perc  <- anova(m_np, m_add)
  lrt_rag   <- anova(m_nr, m_add)
  get_lrt <- function(a) {
    rr <- a[nrow(a), ]
    c(chisq = rr[["LR.stat"]], df = rr[["df"]], p = rr[["Pr(>Chisq)"]])
  }
  lrt_specs <- list(
    list(term = "Perception pathway", contrast = "Additive model versus model without perception pathway",
         res = get_lrt(lrt_perc)),
    list(term = "RAG condition", contrast = "Additive model versus model without RAG",
         res = get_lrt(lrt_rag)),
    list(term = "Perception pathway × RAG", contrast = "Interaction model versus additive model",
         res = get_lrt(lrt_inter))
  )
  for (sp in lrt_specs) {
    lrt_rows[[length(lrt_rows) + 1]] <- data.frame(
      score_dimension = sc, test_term = sp$term, model_comparison = sp$contrast,
      lrt_chisq = round(sp$res["chisq"], 4), df = sp$res["df"],
      p_raw = signif(sp$res["p"], 4),
      comparison_family = "Family 1 (six tests: two dimensions × perception/RAG/interaction)",
      stringsAsFactors = FALSE
    )
  }

  th <- cf[grepl("\\|", rownames(cf)), , drop = FALSE]
  for (i in seq_len(nrow(th))) {
    thresh_rows[[length(thresh_rows) + 1]] <- data.frame(
      score_dimension = sc, threshold = rownames(th)[i],
      alpha = round(th[i, "Estimate"], 4), SE = round(th[i, "Std. Error"], 4),
      stringsAsFactors = FALSE
    )
  }

  vc <- VarCorr(m_full)
  vc_map <- list(sampling_site_id = "Sampling site",
                 observation_block_id = "Observation block",
                 report_id = "Report")
  for (vname in names(vc_map)) {
    v <- vc[[vname]]
    var_rows[[length(var_rows) + 1]] <- data.frame(
      score_dimension = sc, random_effect_level = vc_map[[vname]],
      variance = round(as.numeric(v[1, "(Intercept)"]), 4),
      sd = round(as.numeric(attr(v, "stddev")[1]), 4),
      stringsAsFactors = FALSE
    )
  }

  conv_rows[[length(conv_rows) + 1]] <- data.frame(
    score_dimension = sc,
    convergence_code = m_full$optRes$convergence,
    convergence_message = m_full$optRes$message,
    iterations = m_full$optRes$iterations,
    function_evaluations = m_full$optRes$evaluations["function"],
    gradient_evaluations = m_full$optRes$evaluations["gradient"],
    max_absolute_gradient = signif(max(abs(m_full$gradient)), 4),
    Hessian_dimensions = paste(dim(m_full$Hessian), collapse = "×"),
    stringsAsFactors = FALSE
  )

  freq <- table(factor(dd[[sc]], levels = 1:5))
  sparse_note <- paste0("Score frequencies (1–5): ", paste(freq, collapse = "/"),
                        "; observed scores: ", paste(levels(dd$score), collapse = "-"))
  if (!1 %in% dd[[sc]]) sparse_note <- paste0(sparse_note, "; no score of 1 (sparse categories; one fewer threshold is estimated)")
  diag_notes[[sc]] <- list(freq = freq, sparse = sparse_note)

  mc <- clm(score ~ perception*rag + rater, data = dd)
  nt <- nominal_test(mc); st <- scale_test(mc)
  aux_tests[[sc]] <- list(nominal = nt, scale = st)

  e <- emmeans(m_full, ~ perception*rag)
  grid <- e@grid
  row_idx <- function(p, r) which(grid$perception == p & grid$rag == r)
  v <- function(plus, minus) {
    vec <- rep(0, nrow(grid)); vec[plus] <- 1; vec[minus] <- -1; vec
  }
  contr <- contrast(e, list(
    "B−A" = v(row_idx("MLLM", "with_RAG"), row_idx("MLLM", "without_RAG")),
    "D−C" = v(row_idx("YOLO", "with_RAG"), row_idx("YOLO", "without_RAG")),
    "C−A" = v(row_idx("YOLO", "without_RAG"), row_idx("MLLM", "without_RAG")),
    "D−B" = v(row_idx("YOLO", "with_RAG"), row_idx("MLLM", "with_RAG"))
  ))
  contr_df <- as.data.frame(contr)
  contr_df$label <- contr_df$contrast
  meaning <- c("B−A" = "With versus without RAG within the MLLM pathway",
               "D−C" = "With versus without RAG within the YOLO pathway",
               "C−A" = "YOLO versus MLLM without RAG",
               "D−B" = "YOLO versus MLLM with RAG")
  for (lb in c("B−A", "D−C", "C−A", "D−B")) {
    rr <- contr_df[contr_df$label == lb, ]
    est <- rr$estimate; se <- rr$SE; p_raw <- rr$p.value
    pair_rows[[length(pair_rows) + 1]] <- data.frame(
      score_dimension = sc, contrast_label = lb, contrast_description = meaning[lb],
      log_odds_estimate = round(est, 4), SE = round(se, 4), z = round(rr$z.ratio, 3),
      OR = round(exp(est), 4),
      OR_ci95_lower = round(exp(est - Z95 * se), 4),
      OR_ci95_upper = round(exp(est + Z95 * se), 4),
      p_raw = signif(p_raw, 4),
      comparison_family = "Family 2 (eight contrasts: two dimensions × four prespecified contrasts)",
      stringsAsFactors = FALSE
    )
  }
}

lrt_df <- do.call(rbind, lrt_rows)
lrt_df$p_holm <- signif(p.adjust(lrt_df$p_raw, method = "holm"), 4)

pair_df <- do.call(rbind, pair_rows)
pair_df$p_holm <- signif(p.adjust(pair_df$p_raw, method = "holm"), 4)

coef_df  <- do.call(rbind, coef_rows)
thr_df   <- do.call(rbind, thresh_rows)
var_df   <- do.call(rbind, var_rows)
conv_df  <- do.call(rbind, conv_rows)

lrt_out <- lrt_df[, c("score_dimension", "test_term", "model_comparison", "lrt_chisq", "df",
                      "p_raw", "comparison_family", "p_holm")]

write_parted_csv(OUT_MODEL, list(coef_df))
write_parted_csv(file.path(OUT_DIR, "recommendation_factor_tests.csv"), list(lrt_out))
write_parted_csv(file.path(OUT_DIR, "recommendation_model_thresholds.csv"), list(thr_df))
write_parted_csv(file.path(OUT_DIR, "recommendation_random_effects.csv"), list(var_df))
write_parted_csv(file.path(OUT_DIR, "recommendation_model_convergence.csv"), list(conv_df))

pair_out <- pair_df[, c("score_dimension", "contrast_label", "contrast_description", "log_odds_estimate", "SE", "z",
                        "OR", "OR_ci95_lower", "OR_ci95_upper", "p_raw", "comparison_family", "p_holm")]
write_parted_csv(OUT_PAIR, list(pair_out))

pkgs <- c(ordinal = "ordinal", emmeans = "emmeans")
ver_str <- paste(paste(names(pkgs), sapply(pkgs, function(p) as.character(packageVersion(p))), sep = " "), collapse = "；")

aux_lines <- c()
for (sc in names(score_map)) {
  nt <- as.data.frame(aux_tests[[sc]]$nominal)
  st <- as.data.frame(aux_tests[[sc]]$scale)
  aux_lines <- c(aux_lines, paste0("### ", sc, " — Nominal effects (nominal_test, auxiliary)"))
  aux_lines <- c(aux_lines, "| Term | Df | LRT | p |", "|---|---|---|---|")
  nt <- nt[rownames(nt) != "<none>", , drop = FALSE]
  for (i in seq_len(nrow(nt))) {
    aux_lines <- c(aux_lines, paste0("| ", rownames(nt)[i], " | ", nt$Df[i], " | ",
                                     round(nt$LRT[i], 3), " | ", signif(nt$`Pr(>Chi)`[i], 4), " |"))
  }
  aux_lines <- c(aux_lines, "", paste0("### ", sc, " — Scale effects (scale_test, auxiliary)"))
  aux_lines <- c(aux_lines, "| Term | Df | LRT | p |", "|---|---|---|---|")
  st <- st[rownames(st) != "<none>", , drop = FALSE]
  for (i in seq_len(nrow(st))) {
    aux_lines <- c(aux_lines, paste0("| ", rownames(st)[i], " | ", st$Df[i], " | ",
                                     round(st$LRT[i], 3), " | ", signif(st$`Pr(>Chi)`[i], 4), " |"))
  }
  aux_lines <- c(aux_lines, "")
}

lrt_wide <- lrt_df
concl_lines <- c()
for (sc in names(score_map)) {
  sub <- lrt_wide[lrt_wide$score_dimension == sc, ]
  getp <- function(term) sub$p_holm[sub$test_term == term]
  concl_lines <- c(concl_lines, paste0("- **", sc, "**: perception pathway LRT, Holm-adjusted p=",
                                       getp("Perception pathway"),
                                       "; RAG condition LRT, Holm-adjusted p=",
                                       getp("RAG condition"),
                                       "; interaction LRT, Holm-adjusted p=",
                                       getp("Perception pathway × RAG"), "。"))
}

md <- c(
  "# Ordinal mixed-effects models for recommendation scores: diagnostics",
  "",
  "## 1. Data structure and model specification",
  "",
  "- There are 240 report–rater records from 15 sampling sites, 30 observation blocks, and 120 reports. The models account for pairing across groups within blocks and repeated ratings of each report.",
  "- Cumulative link mixed model with a proportional-odds logit link; original integer scores from 1 to 5 are treated as ordered factors.",
  "- Formula: `score ~ perception * rag + rater + (1|sampling_site_id) + (1|observation_block_id) + (1|report_id)`.",
  "- Reference levels: MLLM for perception (YOLO is the other level), without RAG (with RAG is the other level), and Rater1 (Rater2 is the other level).",
  "- Software: R 4.5.3; ", ver_str, "。",
  "",
  "## 2. Convergence and random intercept variances",
  "",
  "| Score dimension | Convergence code | Message | Iterations | Maximum absolute gradient | Hessian dimensions | Site variance | Block variance | Report variance |",
  "|---|---|---|---|---|---|---|---|---|"
)
for (sc in names(score_map)) {
  cv <- conv_df[conv_df$score_dimension == sc, ]
  vv <- var_df[var_df$score_dimension == sc, ]
  md <- c(md, paste0("| ", sc, " | ", cv$convergence_code, " | ", cv$convergence_message, " | ",
                     cv$iterations, " | ", cv$max_absolute_gradient, " | ", cv$Hessian_dimensions, " | ",
                     vv$variance[vv$random_effect_level == "Sampling site"], " | ",
                     vv$variance[vv$random_effect_level == "Observation block"], " | ",
                     vv$variance[vv$random_effect_level == "Report"], " |"))
}
md <- c(md, "",
  "The model retains random intercepts for sampling site, observation block, and report. Convergence codes, maximum absolute gradients, and variance estimates are listed above.",
  "",
  "## 3. Sparse score categories",
  "")
for (sc in names(score_map)) {
  md <- c(md, paste0("- **", sc, "**：", diag_notes[[sc]]$sparse, "。"))
}
md <- c(md,
  "- Observed logical coherence scores range from 2 to 5, with no score of 1. The CLMM therefore estimates thresholds 2|3, 3|4, and 4|5; threshold 1|2 is not identifiable. This sparse category is documented; no categories were merged or observations removed.",
  "",
  "## 4. Proportional-odds assumption checks",
  "",
  "The ordinal package provides `nominal_test()` and `scale_test()` for `clm`, but not `clmm`. A `clm` with the same fixed effects, omitting random effects, is fitted as an auxiliary check of proportional odds (parallel slopes). Its LRT results are shown below. Primary inference remains based on the CLMM; these auxiliary tests do not determine main-effect significance.",
  "",
  aux_lines,
  "In the archived analysis, nominal and scale tests for perception, RAG, and their interaction were nonsignificant (p>0.05), providing no evidence against proportional odds. Some rater tests were significant, suggesting differences in rating scales or intercepts; rater is included as a fixed effect.",
  "",
  "## 5. Overall factor tests (family 1: six Holm-adjusted tests)",
  "",
  "Nested-model LRTs compare interaction versus additive models, additive versus no-perception models, and additive versus no-RAG models. Family 1 comprises six two-sided tests across the two dimensions, with Holm adjustment.",
  "",
  concl_lines,
  "",
  "In the full interaction model, individual perception and RAG coefficients are conditional on reference levels, even when the interaction is nonsignificant. They should not be interpreted as overall factor effects; use the LRTs above for overall tests. Nonsignificance does not establish equivalence.",
  "",
  "## 6. Prespecified paired contrasts (family 2: eight Holm-adjusted contrasts)",
  "",
  "The interaction CLMM is used with emmeans to estimate log-odds contrasts: B−A and D−C compare with versus without RAG within each pathway; C−A and D−B compare YOLO versus MLLM without and with RAG. OR=exp(estimate); 95% intervals exponentiate the link-scale estimate ±1.96SE. Family 2 comprises eight two-sided Holm-adjusted contrasts. See `recommendation_pairwise_contrasts.csv` for raw and adjusted p values.",
  "",
  "## 7. Bootstrap settings for rater agreement",
  "",
  "- Linear weighted Cohen's kappa is primary; quadratic weighted kappa is supplementary. The 95% percentile intervals use 5,000 sampling-site cluster bootstrap replicates with seed 20261122. Outputs from rater_sensitivity.R are saved in the rater_sensitivity directory.",
  "",
  "## 8. Model scope and limitations",
  "",
  "- The primary model includes sampling site, observation block, and report random intercepts to account for paired reports and repeated ratings.",
  "- The auxiliary clm assumption check is used because ordinal does not provide the corresponding clmm methods; it does not change the primary model or inference.",
  "- Logical coherence has no score of 1. This sparse category is recorded without merging score levels.",
  "- Nonsignificant results do not establish equivalence; no non-inferiority or equivalence inference is made."
)

writeLines(md, OUT_DIAG, useBytes = TRUE)

cat("== Overall LRTs (family 1) ==\n"); print(lrt_out)
cat("\n== Paired contrasts (family 2) ==\n"); print(pair_out)
cat("\nOutputs:\n", OUT_MODEL, "\n", OUT_PAIR, "\n", OUT_DIAG, "\n", sep = "")
