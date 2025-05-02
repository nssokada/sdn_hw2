functions {
    real hybrid(int num_trials, array[] int action1, array[] int s2, array[] int action2,
        array[] int reward, real alpha1, real alpha2, real lmbd, real beta1, 
        real beta2, real w, real p) {

        real log_lik;
        array[2] real q;
        array[2, 2] real v;

        // Initializing values
        for (i in 1:2)
            q[i] = 0;
        for (i in 1:2)
            for (j in 1:2)
                v[i, j] = 0;

        log_lik = 0;

        for (t in 1:num_trials) {
            real x1;
            real x2;
            x1 = // Model-based value
                w*0.4*(max(v[2]) - max(v[1])) +
                // Model-free value
                (1 - w)*(q[2] - q[1]);
            // Perseveration
            if (t > 1) {
                if (action1[t - 1] == 2)
                    x1 += p;
                else
                    x1 -= p;
            }
            // Exploration
            x1 *= beta1;
            // First stage choice
            if (action1[t] == 2)
                log_lik += log_inv_logit(x1);
            else
                log_lik += log1m_inv_logit(x1);

            // Second stage choice
            x2 = beta2*(v[s2[t], 2] - v[s2[t], 1]);
            if (action2[t] == 2)
                log_lik += log_inv_logit(x2);
            else
                log_lik += log1m_inv_logit(x2);

            // Learning
            q[action1[t]] += alpha1*(v[s2[t], action2[t]] - q[action1[t]]) +
                alpha1*lmbd*(reward[t] - v[s2[t], action2[t]]);
            v[s2[t], action2[t]] += alpha2*(reward[t] - v[s2[t], action2[t]]);
        }
        return log_lik;
    }
}
data {
    int<lower=0> N; // Number of participants
    int<lower=0> maxtrials;
    // Number of trials (can be < maxtrials if participant missed some)
    array[N] int<lower=1, upper=maxtrials> num_trials;
    array[N, maxtrials] int<lower=1, upper=2> action1; // First stage actions
    array[N, maxtrials] int<lower=1, upper=2> action2; // Second stage actions
    array[N, maxtrials] int<lower=1, upper=2> s2; // Second stage states
    array[N, maxtrials] int<lower=0, upper=1> reward; // Rewards

    //fixed params flags
    int<lower=0, upper=1> fix_alpha1;
    int<lower=0, upper=1> fix_alpha2;
    int<lower=0, upper=1> fix_lmbd;
    int<lower=0, upper=1> fix_beta1;
    int<lower=0, upper=1> fix_beta2;
    int<lower=0, upper=1> fix_w;
    int<lower=0, upper=1> fix_p;

    //the values we might pass
    real<lower=0, upper=1> alpha1;
    real<lower=0, upper=1> alpha2;
    real<lower=0, upper=1> lmbd;
    real<lower=0> beta1;
    real<lower=0> beta2;
    real<lower=0, upper=1> p;
    real<lower=0, upper=1> w;
}


parameters {
    array[fix_alpha1 ? 0 : 1] real<lower=0, upper=1> alpha1_free;
    array[fix_alpha2 ? 0 : 1] real<lower=0, upper=1> alpha2_free;
    array[fix_lmbd ? 0 : 1] real<lower=0, upper=1> lmbd_free;
    array[fix_beta1 ? 0 : 1] real<lower=0> beta1_free;
    array[fix_beta2 ? 0 : 1] real<lower=0> beta2_free;
    array[fix_p ? 0 : 1] real<lower=0, upper=1> p_free;

    array[fix_w ? 0 : N] real<lower=0, upper=1> w_free;
}

transformed parameters {
  real<lower=0, upper=1> alpha1_local = fix_alpha1 ? alpha1 : alpha1_free[1];
  real<lower=0, upper=1> alpha2_local = fix_alpha2 ? alpha2 : alpha2_free[1];
  real<lower=0, upper=1> lmbd_local = fix_lmbd ? lmbd : lmbd_free[1];
  real<lower=0> beta1_local = fix_beta1 ? beta1 : beta1_free[1];
  real<lower=0> beta2_local = fix_beta2 ? beta2 : beta2_free[1];
  real<lower=0, upper=1> p_local = fix_p ? p : p_free[1];
  
  vector<lower=0, upper=1>[N] w_local;
  
  for (i in 1:N) {
    if (fix_w) {
      w_local[i] = w;
    } else {
      w_local[i] = w_free[i];
    }
  }
}


model {
    for (i in 1:N) {
        target += hybrid(num_trials[i], action1[i], s2[i], action2[i], reward[i], 
            alpha1_local, alpha2_local, lmbd_local, beta1_local, beta2_local, 
            w_local[i], p_local);
    }
}