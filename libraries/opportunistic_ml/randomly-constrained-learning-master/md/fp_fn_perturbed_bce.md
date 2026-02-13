To understand this loss, let $\sigma$ be the sigmoid function
$$
\sigma(t)=\frac1{1+e^{-t}}
$$
and define
$$
\varphi(t,\gamma)=
\begin{cases}\displaylines{
-\frac{\sigma(t)^\gamma}\gamma\text{ if }\gamma>0\text{ and}\\\\
-\log(\sigma(t))\text{ otherwise.}}
\end{cases}.
$$
The loss considered is then defined piecewise with respect to the target:
$$
L(\rho,y,(\alpha,\beta))=
\begin{cases}
\varphi(\rho,\alpha)\text{ if }y=\boldsymbol+\text{ and}\\\\
\varphi(-\rho,\beta)\text{ otherwise.}
\end{cases}
$$
If the NN architecture already applies a sigmoid activation on the last
layer, then one could obtain an equivalent NN arhitecture by substituting
the same loss but without first applying $\sigma$ to the NN output.