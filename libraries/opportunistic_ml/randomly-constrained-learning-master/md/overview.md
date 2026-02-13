# Randomly constrained learning utilities

## Problem formulation

We consider the problem of classification where performance is measured according to dynamically fluctuating user constraints.
- We use adam to train a NN, stochastically nudging the function's output to be larger or smaller according to the class label.
- After training, a user makes a binary inference by applying some cutoff for a float output

### Overview of imbalanced binary classification
#### Introducton
Let $x$ and $y$ be random variables, where we suppose that the value of $x\in\mathcal X$ gives an indication of likely possible values of $y\in\mathcal Y$.
In the case of binary classification there are only two classes and we write $\mathcal Y=\boldsymbol 2$ for the two class labels $\boldsymbol+$ and $\boldsymbol-$.
The aim is to construct a model to predict which of these two possible values $y$ is based on the value of $x$.
Of particular interest are cases where $x$ belongs to a majority or a minority class with high and low probability respectively.
When discussing such problems we let $y=+$ be the minority class.
In the supervised learning paradigm, models are constructed based on a training set of pairs of inputs and labels $(x_{\verb|train|},y_{\verb|train|})\in\mathcal X^{n_{\verb|train|}}\times\boldsymbol 2^{n_{\verb|train|}}$ and subsequently evaluated against a testing set $(x_{\verb|test|},y_{\verb|test|})\in\mathcal X^{n_{\verb|test|}}\times\boldsymbol 2^{n_{\verb|test|}}$.

We focus on training neural networks (NNs) with gradient descent.

### Discrete performance metrics for binary classifiers
When evaluating a model's performance, each binary prediction and label $(\widetilde\upsilon_i,\upsilon_i)$ is either a true positive $(+,+)$, a false positive $(+,-)$, a false negative $(-,+)$ or a true negative $(-,-)$.

When a batch of these observations are made, denote the counts of these corresponding cases by $TP$, $FP$, $FN$ and $TN$ respectively.
Standard associated statistics are
- the $\textbf{sample(/batch) precision}=\frac{TP}{TP+FP}$,
- the $\textbf{sample recall}=\frac{TP}{TP+FN}$ and
- the $\textbf{sample accuracy}=\frac{TP+TN}{TP+FP+TN+FN}$.

These quantities are easily understood by someone working on a wide range of possible problems.
In the context of spam filtering, when differentiating between ham and spam, we don't necessarily mind if we sometimes classify ham as spam, but we really should try to insist that spam never makes it through.

You could achieve this by blocking everything, so we can't merely focus on preventing false negatives.
We should try and insist instead on some combination of accuracy and minimising false negatives.

One may also worry about this sort of problem when testing for rare diseases.
Not running any test at all may be highly accurate and possibly preferable to a test with too many false positives.

If we assume that the data is drawn from fixed distributions, then these sample statistics approximate true rates associated to some model with parameter $\boldsymbol\theta$.
Write $\mathbb P_{\boldsymbol\theta}$ for the probabilities of events for such a fixed model.
- The $\textbf{precision}=\mathbb P_{\boldsymbol\theta}(+|\text{We predicted }+)$, and
- the $\textbf{recall}=\mathbb P_{\boldsymbol\theta}(\text{we predicted }+|+)$.
