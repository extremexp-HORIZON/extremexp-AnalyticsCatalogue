# Costs and tails applied to the University of New South Wales
- Here we train without and then with resampliung
- Let $\sigma$ be the sigmoid function (which for binary classification can be intuited as mapping log relative likelihoods to probabilities) and define
$$
  \varphi(t,\gamma)=
  \begin{cases}
  -\frac{\sigma(t)^\gamma)}\gamma\text{ if }\gamma>0\text{ and}\\
  -\log(t)\text{ otherwise.}
  \end{cases}
$$
- Train a NN to output $\rho$ according to loss
$$
L(\rho,y,(\alpha,\beta))=
\begin{cases}
\varphi(t,\alpha)\text{ if }y=\boldsymbol+\text{ and}\\
\varphi(-t,\beta)\text{ otherwise.}
\end{cases}
$$
- This can be seen up to a constant as a family of perturbations of BCE (which is the case $\alpha=\beta=0$), but has similar asymptotics.
- Will try the originally intended loss in another notebook.
- Referenced paper - $\alpha$ and $\beta$ should correspond to different tails on the distributions of costs for false negatives and positives.
