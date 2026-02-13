"""
Loss
"""
from jax.nn import sigmoid
from jax.numpy import log
from pathlib import Path

from optax.losses import sigmoid_binary_cross_entropy

def fp_fn_perturbed_bce(pred, targ, param):
    """
    Parametric loss function for NN training parameterised by a parameter
    param=(alpha,beta) where {alpha|beta} define the asymptotics of how
    loss varies with pred when the target is {+|-} respectively
    Typically this isn't called directly, but passed to jax.grad.

    Parameters
    ----------
    pred : ndarray[float]
        Continuous model predictions
    targ : ndarray[bool]
        Target labels
    param : tuple[float,float]
        Parameter controlling tails of the loss for fixed target

    Returns
    ----------
    err : ndarray
        Loss function value
    """
    a, b = param
    _l_pos = (lambda r: (1 - sigmoid(r) ** a) / a) if a else\
             (lambda r: -log(sigmoid(r)))  # loss if y=+
    _l_neg = (lambda r: (1 - sigmoid(-r) ** b) / b) if b else\
             (lambda r: -log(sigmoid(-r)))  # lloss if y=-
    return targ * _l_pos(pred) + (1 - targ) * _l_neg(pred)

losses = {'fp_fn_perturbed_bce':fp_fn_perturbed_bce,
          'sigmoid_binary_cross_entropy':sigmoid_binary_cross_entropy}


_notes_header ="""
Notes
-----
"""
lines = fp_fn_perturbed_bce.__doc__.split('\n')
n_leading_spaces = min([len(ln)-len(ln.lstrip()) for ln in lines if ln])
fp_fn_perturbed_bce.__doc__ = '\n'.join([ln[n_leading_spaces:] for ln in lines])
fp_fn_perturbed_bce.__doc__ +=_notes_header +\
                            (Path(__file__).parent.parent\
                             / 'md' / 'fp_fn_perturbed_bce.md').read_text()