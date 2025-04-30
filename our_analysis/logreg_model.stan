data {
    int M; // number of participants
    int N; // number of trials per participant
    int K; // number of predictors

    array[M, N] int<lower=0, upper=1> y; // stay/switch outcomes
    array[M] matrix[N, K] x;             // predictors for each participant
    array[M] int<lower=0, upper=1> condition; // binary condition (e.g., story=1, abstract=0)
}

parameters {
    // Individual-level logistic regression weights
    array[M] vector[K] coefs;

    // Group-level priors
    cholesky_factor_corr[K] L_Omega;
    vector<lower=0>[K] tau;
    vector[K] beta0;
    vector[K] condition_beta;
}

transformed parameters {
    matrix[K, K] Sigma;
    Sigma = diag_pre_multiply(tau, L_Omega);
    Sigma = Sigma * Sigma'; // full covariance matrix
}

model {
    // Priors
    tau ~ cauchy(0, 1);
    beta0 ~ normal(0, 5);
    condition_beta ~ cauchy(0, 1);
    L_Omega ~ lkj_corr_cholesky(2);

    // Hierarchical regression with condition-specific means
    for (p in 1:M) {
        coefs[p] ~ multi_student_t(4, beta0 + condition[p] * condition_beta, Sigma);
        y[p] ~ bernoulli_logit(x[p] * coefs[p]);
    }
}
