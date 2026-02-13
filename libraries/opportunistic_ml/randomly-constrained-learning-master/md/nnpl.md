The development of this class is motivated primarily by the problem of
training NNs for binary classification.
We are interested in investigating appropriate loss and resampling
scheme selection depending on a the requirements a user makes of a
trained model.

Such loss functions may either
- have a direct theoretical justification as in the work
 (https://link.springer.com/article/10.1007/s10994-024-06634-8)
of Komisarenko and Kull,
- Be an ansatz with similar asymptotic properties to a theoretically
derived losses in the above paper, or
- be an arbitrary surrogate/ansatz motivated by some other argument.
