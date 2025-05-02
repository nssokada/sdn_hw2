Seven free parameters in the hybrid model: $α^i_1$, $α^i_2$, $λ^i$, $β^i_1$, $β^i_2$, $w^i$ and $p^i$

The i-th superscript indicates the parameter for the i-th participant. 

(I am not used to github so plz edit the format if needed to)

1. $α^i_1$ - the first-stage learning rate (0 $\leq$ $α_1$ $\leq$ 1): controls how much the model-free values of the first-stage actions are updated based on the outcomes of a trial.

2. $α^i_2$ - the second-stage learning rate (0 $\leq$ $α_2$ $\leq$ 1): controls how much the model-free values of the second-stage actions are updated based on the reward prediction errors.

3. $λ^i$ - the eligibility trace parameter (0 $\leq$ $λ$ $\leq$ 1): modulates how much the reward prediction error from the second stage affects the first-stage model-free values.

4. $β^i_1$ - the first-stage’s inverse temperature parameter, which determines the exploration rate (random vs consistent) at the first stage.  

5. $β^i_2$ - the second-stage’s inverse temperature parameter, which determines the exploration rate (random vs consistent) at the second stage. 

6. $w^i$ - the model-based weight (0 $\leq$ $w$ $\leq$ 1): determines the relative contribution of model-based versus model-free value computations at the first stage.

7. $p^i$ - the perseveration parameter, which adds a tendency to repeat the previous trial’s first-stage action in the next trial. 
