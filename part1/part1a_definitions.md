Seven free parameters in the hybrid model: $α^i_1$, $α^i_2$, $λ^i$, $β^i_1$, $β^i_2$, $w^i$ and $p^i$

The $i$th superscript indicates the parameter for the $i$th participant. 

(I am not used to github so plz edit the format if needed to)

1. $α^i_1$ - the first-stage learning rate (0 $\leq$ $α_1$ $\leq$ 1)
2. $α^i_2$ - the second-stage learning rate (0 $\leq$ $α_2$ $\leq$ 1)
3. $λ^i$ - the eligibility trace parameter (0 $\leq$ $λ$ $\leq$ 1)
4. $β^i_1$ - the first-stage’s inverse temperature parameter, which determines the exploration rate at the first stage. 
5. $β^i_2$ - the second-stage’s inverse temperature parameter, which determines the exploration rate at the second stage. 
6. $w^i$ - the model-based weight (0 $\leq$ $w$ $\leq$ 1)
7. $p^i$ - the perseveration parameter, which adds a tendency to repeat the previous trial’s first-stage action in the next trial
