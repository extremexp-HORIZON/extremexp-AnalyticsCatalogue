# Scalable ML



## Getting started

To utilize the contents of this repository we intend the usage to be as follows.
The usage is to address the data privacy/security concerns when performing operations using private data on the extremeXP framework and execution engine.

We have included functions that can be used to overcome this when using a public or 3rd party execution engine.

We use the fact that local neural encoders will preserve model performance when used in conjunctions with downstream training.
However different datasets with different data tasks will require different encoding mechanisms.
This is where we step in with a modification to existing results.

We can employ the training of an autoencoder module that will be trained locally, however we need to acknowledge that different tasktypes require different architectures which can have very different training costs.
Thus we want to utilize certain transfer learning methods,  by using pre-trained(transfer learning) models as shallow encoders, there is sufficient evidence to preserve the privacy of the data.

Through various neural methods, we have ensured that the operations are all reversible, however due to the fact that we assume the execution engine will be much more powerful and performing the full experiment then the autoencoder that is only neccesary is the encoder.
Thus we have created encoders whereby the operations are fully reversible, meaning that the decoder definitely exists however we do not provide it as it is not needed for use.

We have included a jupyter notebook, testing.ipynb, to illustrate how the code can be used in conjunction with the cyber security dataset.
We have illustrated the usage through two different model types MLPs, and RNNs.
However, it should not be constrained to this.

We have also included a very crude functionality on the encoder cost in terms of memory.


