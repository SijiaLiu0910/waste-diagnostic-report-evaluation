# Rater-specific models, rater-by-condition interactions, and weighted kappa.
suppressPackageStartupMessages({library(ordinal); library(emmeans)})
.script <- local({a <- commandArgs(trailingOnly = FALSE)
                  f <- sub("^--file=", "", a[grepl("^--file=", a)])
                  if (length(f)) f[1] else ""})
root <- normalizePath(file.path(if (nzchar(.script)) dirname(.script) else ".",
                                "..", ".."), mustWork = FALSE)
input <- file.path(root, 'results/evaluation/report_quality/ratings_long.csv')
out <- file.path(root, 'results/recomputed/statistics/rater_sensitivity')
dir.create(out, recursive=TRUE, showWarnings=FALSE)
set.seed(20261122)
boot_n <- 5000L
d <- read.csv(input, fileEncoding='UTF-8-BOM', check.names=FALSE)
stopifnot(nrow(d)==240L, length(unique(d$report_id))==120L,
          all(table(d$report_id)==2L), length(unique(d$sampling_site_id))==15L)
d$P <- factor(d$perception_pathway, levels=c('MLLM','YOLO'))
d$R <- factor(d$rag_condition, levels=c('without_RAG','with_RAG'))
d$V <- factor(d$rater, levels=c('Rater1','Rater2'))
scores <- c('reliability','logical_coherence')
write_csv <- function(x, name) {
  con <- file(file.path(out, name), open='wb')
  on.exit(close(con))
  writeBin(as.raw(c(0xef,0xbb,0xbf)), con)
  write.table(x, con, sep=',', row.names=FALSE, na='NA', qmethod='double')
}
ci <- function(x) unname(quantile(x[is.finite(x)], c(.025,.975)))
des <- list(); diag <- list(); main <- list(); pairs <- list(); tests <- list()
full_models <- list()
record_diag <- function(m, score, label) {
  ev <- eigen(m$Hessian, symmetric=TRUE, only.values=TRUE)$values
  data.frame(score_dimension=score,model=label,n_ratings=nobs(m),
    convergence_code=m$optRes$convergence,max_absolute_gradient=max(abs(m$gradient)),
    Hessian_min_eigenvalue=min(ev),Hessian_condition_number=max(ev)/min(ev),
    logLik=as.numeric(logLik(m)),AIC=AIC(m),
    random_effect_variances=paste(names(VarCorr(m)),
      vapply(VarCorr(m), function(z) as.numeric(z[1,1]), numeric(1)),
      sep='=',collapse=';'),check.names=FALSE)
}
extract_contrasts <- function(m, score, label) {
  e <- emmeans(m, ~P*R, mode='latent')
  g <- as.data.frame(e)
  names4 <- ifelse(g$P=='MLLM',ifelse(g$R=='without_RAG','A','B'),
                   ifelse(g$R=='without_RAG','C','D'))
  weight <- function(w) unname(w[names4])
  eff <- contrast(e, list(
    'YOLO versus MLLM with RAG status held constant'=weight(c(A=-.5,B=-.5,C=.5,D=.5)),
    'With versus without RAG with perception pathway held constant'=weight(c(A=-.5,B=.5,C=-.5,D=.5)),
    'Difference between RAG-related changes in the YOLO and MLLM pathways'=weight(c(A=1,B=-1,C=-1,D=1))),adjust='none')
  pp <- contrast(e, list(
    'B−A'=weight(c(A=-1,B=1,C=0,D=0)),
    'D−C'=weight(c(A=0,B=0,C=-1,D=1)),
    'C−A'=weight(c(A=-1,B=0,C=1,D=0)),
    'D−B'=weight(c(A=0,B=-1,C=0,D=1))),adjust='none')
  cv <- function(obj) {
    z <- as.data.frame(summary(obj, infer=c(FALSE,TRUE),adjust='none'))
    data.frame(score_dimension=score,rater=label,contrast=z$contrast,
      logOR=z$estimate,SE=z$SE,OR=exp(z$estimate),
      OR_ci95_lower=exp(z$estimate-qnorm(.975)*z$SE),
      OR_ci95_upper=exp(z$estimate+qnorm(.975)*z$SE),
      p_raw=z$p.value,check.names=FALSE)
  }
  list(main=cv(eff), pairs=cv(pp))
}
for(sc in scores) {
  d$score <- ordered(d[[sc]])
  base <- clmm(score~P*R+V+(1|sampling_site_id)+(1|observation_block_id)+(1|report_id),data=d,Hess=TRUE)
  full <- clmm(score~P*R*V+(1|sampling_site_id)+(1|observation_block_id)+(1|report_id),data=d,Hess=TRUE)
  full_models[[sc]] <- list(primary_refit=base,rater_interactions=full)
  diag[[length(diag)+1L]] <- record_diag(base,sc,'Refitted pooled primary model')
  diag[[length(diag)+1L]] <- record_diag(full,sc,'Pooled model with rater-by-condition interactions')
  lr <- anova(base,full)
  tests[[length(tests)+1L]] <- data.frame(score_dimension=sc,
    test='Joint exploratory LRT for the additional V:P, V:R, and V:P:R interactions',
    LRT=lr$LR.stat[2],df=lr$df[2],p_raw=lr[['Pr(>Chisq)']][2],
    AIC_primary=AIC(base),AIC_interaction=AIC(full),check.names=FALSE)
  cm <- extract_contrasts(base,sc,'Pooled primary model')
  main[[length(main)+1L]] <- cm$main
  for(vr in levels(d$V)) {
    s <- droplevels(subset(d,V==vr))
    for(gr in c('A','B','C','D')) {
      vals <- s[s$group==gr,sc]
      des[[length(des)+1L]] <- data.frame(score_dimension=sc,rater=vr,group=gr,
        n_reports=length(vals),mean=mean(vals),sd=sd(vals),median=median(vals),
        Q1=unname(quantile(vals,.25)),Q3=unname(quantile(vals,.75)),check.names=FALSE)
    }
    m <- clmm(score~P*R+(1|sampling_site_id)+(1|observation_block_id),data=s,Hess=TRUE)
    full_models[[sc]][[paste0('rater_',vr)]] <- m
    diag[[length(diag)+1L]] <- record_diag(m,sc,paste0('Only ',vr,' ratings; retain sampling site and observation block random intercepts'))
    cc <- extract_contrasts(m,sc,vr)
    main[[length(main)+1L]] <- cc$main
    pairs[[length(pairs)+1L]] <- cc$pairs
  }
}
desc <- do.call(rbind,des); diagnostics <- do.call(rbind,diag)
mm <- do.call(rbind,main); pp <- do.call(rbind,pairs); tt <- do.call(rbind,tests)
mm$p_holm_18_exploratory_contrasts <- p.adjust(mm$p_raw,'holm')
pp$p_holm_16_exploratory_contrasts <- p.adjust(pp$p_raw,'holm')
tt$p_holm_two_joint_tests <- p.adjust(tt$p_raw,'holm')
write_csv(desc,'rater_group_descriptives.csv')
write_csv(mm,'rater_factor_contrasts.csv')
write_csv(pp,'rater_pairwise_contrasts.csv')
write_csv(tt,'rater_condition_interaction_tests.csv')
write_csv(diagnostics,'sensitivity_model_diagnostics.csv')

wide <- reshape(d[,c('report_id','sampling_site_id','observation_block_id','rater',scores)],
  idvar=c('report_id','sampling_site_id','observation_block_id'),timevar='rater',direction='wide')
stopifnot(nrow(wide)==120L)
kappa <- function(a,b,power) {
  tab <- table(factor(a,levels=1:5),factor(b,levels=1:5))
  distance <- (abs(outer(1:5,1:5,'-'))/4)^power
  expected <- outer(rowSums(tab),colSums(tab))/sum(tab)
  den <- sum(distance*expected)
  if(den==0) return(NA_real_)
  1-sum(distance*tab)/den
}
points <- sort(unique(wide$sampling_site_id))
boot_idx <- replicate(boot_n,unlist(lapply(sample(points,length(points),replace=TRUE),
  function(pt) which(wide$sampling_site_id==pt))),simplify=FALSE)
kk <- list()
for(sc in scores) {
  a <- wide[[paste0(sc,'.Rater1')]];b <- wide[[paste0(sc,'.Rater2')]]
  for(power in 1:2) {
    bv <- vapply(boot_idx,function(ix) kappa(a[ix],b[ix],power),numeric(1))
    bounds <- ci(bv)
    kk[[length(kk)+1L]] <- data.frame(score_dimension=sc,
      weighting=ifelse(power==1,'Linear; prespecified primary measure','Quadratic; supplementary comparison'),
      kappa=kappa(a,b,power),ci95_lower=bounds[1],ci95_upper=bounds[2],
      exact_agreement=mean(a==b),agreement_within_one_point=mean(abs(a-b)<=1),
      bootstrap_replicates=boot_n,valid_bootstrap_replicates=sum(is.finite(bv)),seed=20261122,
      check.names=FALSE)
  }
}
kk <- do.call(rbind,kk)
write_csv(kk,'weighted_kappa_comparisons.csv')
capture.output(sessionInfo(),file=file.path(out,'R_sessionInfo.txt'))

summary <- c(
  'Rater sensitivity analysis and rating agreement',
  paste('Input file MD5:',unname(tools::md5sum(input))),
  'Models: pooled ratings, rater-by-condition interactions, and separate ordinal mixed-effects models for each rater.',
  'Rater-specific models retain sampling site and observation block random intercepts and allow different score thresholds.',
  'Perception contrast=(C+D-A-B)/2; RAG contrast=(B+D-A-C)/2; interaction=D-C-B+A, all on the latent log-odds scale.',
  'Multiple comparisons: separate Holm families for two joint interaction tests, 16 rater-specific pairwise contrasts, and 18 factorial contrasts.',
  'OR intervals are 95% Wald intervals; tests retain their respective adjustment families.',
  'Rating agreement: linear weighted kappa is primary and quadratic weighted kappa supplementary; 5,000 sampling-site cluster bootstrap replicates with percentile intervals and seed 20261122.',
  '\nJoint interaction tests:',capture.output(print(tt,row.names=FALSE)),
  '\nFactorial contrasts:',capture.output(print(mm,row.names=FALSE)),
  '\nAgreement:',capture.output(print(kk,row.names=FALSE)),
  '\nModel diagnostics:',capture.output(print(diagnostics,row.names=FALSE)))
writeLines(summary,file.path(out,'sensitivity_analysis.txt'),useBytes=TRUE)
cat('OUTPUT:',out,'\n')
print(tt,row.names=FALSE)
print(mm[,c('score_dimension','rater','contrast','OR','OR_ci95_lower','OR_ci95_upper','p_holm_18_exploratory_contrasts')],row.names=FALSE)
print(kk,row.names=FALSE)
