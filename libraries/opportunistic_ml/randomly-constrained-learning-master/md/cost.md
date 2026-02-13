## Randomly constrained learning
When a user uses a trained model to make a prediction, they do so in the understanding that misclassifications are worse than correct predictions.
Commonly, as, for example, in the case of threat detection in cybersecurity, different possible types of misclassifications in general have quite different consequences.
Not only do these **costs** differ, but they may fluctuate dynamically according to a user's varying requirements.

Write $x$ for the feature a model uses to make a class prediction $\hat y(x)$ of the true label $y$.

### Costs

It is always possible to avoid one type of error by only ever predicting one class, so it is insufficient to measure a classifier's performance by simply looking at either of these two error rates in isolation, and similarly one may always guarantee either perfect accuracy or perfect precision.
In order to arrive at a well defined objective for either benchmarking or training, one must place a constraint on the relationship between the costs of different error types.
This constraint must capture the payoffs a user must first determine between the consequences of different error types.

Typically a user has some knowledge of the expected costs of each type of misclassification, and in this work we focus on an optimisation problem where varying costs of false negatives are typically much larger and/or have fatter tail risks than those of false positives.

We focus on the case where the classifier is designed to identify rare but important events, where "false alarm" false positive errors are more tolerable than missing an actual event of interest.

We assume the following:
- The costs $c_+$ and $c_-$ of a false negative or a false positive are random variables
- Given a sampled feature $x=x'$ the user wishes to minimise the *expected cost* of the prediction made by the model:
\begin{align*}
\mathbb E(\text{cost})=&\int_{x':\hat y(x')=+}\mathbb E(c_-|x=x',\hat y=+)f_{x\times y}(x',-)\text{d} x'+\\
&\int_{x':\hat y(x')=-}\mathbb E(c_+|x=x',\hat y=+)f_{x\times y}(x',+)\text{d} x',
\end{align*}
where $f_z$ denotes the probability density function of a random variable $z$.
- At training time some prior knowledge of the uncertainties may be known, and the user can specify the actual consequences at the actual prediction step.

The distributions of $c_+$ and $c_-$ thus define the risk profile for a given training scenario.

### Associating objective functions to risk profiles
The 2025 work of Komisarenko and Kull
[Cost‑sensitive classification with cost uncertainty: do we
need surrogate losses
](https://link.springer.com/article/10.1007/s10994-024-06634-8)
associates a unique loss function to a given user's cost uncertainties.
This loss function is a natural benchmark for a user wishing to check how a model will perform in production under their assumptions, but is (for nonpathological cost uncertainties) also a differentiable map as required for NN training.

Oe may observe that the asymptotic properties of the losses derived in this paper correspond to the profile of the tails of the distributions of $c_+$ and $c_-$.

### A candidate family of losses
Here models are trained where  where costs of errors will be known immediately
before a prediction is made, but such precise information is unavailable at
training time.  This method can be applied as long as the user can estimate the
distributions of the costs to be encountered

To test the appropriateness of losses over a range of different scenarios, we choose a parametric generalisation of binary cross entropy to allow for a range of asymptotic behaviours.

The binary cross entropy loss in the expected cost formulation can be derived as a limiting case where the uncertainty in the costs of both classes diverges.
Since the logarithm itself can be expressed as the limit
$$
\log(x)=\lim_{t\rightarrow0}\frac{x^t-1}t,
$$
we choose the family of functions parameterised by two numbers $\alpha$ and $\beta$ such that the loss for false negatives and false positives is instead given by this power.
If $\alpha=\beta$ this loss is equivalent to an $L^p$ or binary cross entropy loss, but experimentally we find that distinct values of these parameters often produce the best outcomes.
From the theoretical perspective above, as the values of $\alpha$ and $\beta$ increase, the loss can be associated with cost distributions with increasingly thin tails.

### Experimental method

For simplicity we benchmark the results over a range of fixed
ratios - this is in slight to one of the important points of the above paper,
that the appropriate statistic against which to benchmark a model is the
sample expected cost on a dataset.

Strictly speaking, these losses correspond to differing *tail errors* associated with of misclassifications, rather than for fixed costs.
For a specific real world scenario, one may derive an appropriate statistic to be minimised when testing models before production use.

Nonetheless, these experiments demonstrate that good choices of losses and
resampling schemes are heavily contingent upon the user's requirements for a
model.