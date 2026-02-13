from typing import Any
from collections.abc import Iterable, Callable

from flax.nnx import Rngs, Dropout, Linear, LayerNorm
from rcl.comp import Parallel, Sequential, Fun

from jax import Array
from jax.nn import tanh, relu, softmax
from jax.numpy import squeeze, array, concat, nan_to_num, isnan, inf


def CompLin(*a, **k):
  return Fun(Linear(*a, **k))


@Fun
def nan_to_0(arr: Array)->Array:
    return nan_to_num(arr, nan = 0, posinf = 0, neginf = 0)


@Fun
def sm_bag(arr:Array)->Array:
    print(arr.shape)
    return softmax(arr, axis = -2)


@Fun
def expected_embedding(emb, probs:Array)->Array:
    return (probs * emb).sum(axis = -2)


@Fun
def Id(*x:object)->object:
    return x if len(x) != 1 else x[0]


def mlp(k, dims: Iterable[int], dos: Iterable[float]|float|None = None,
        logf: Callable = print, act: Callable = relu)->Sequential:
    if dos is None or isinstance(dos, float):
        dos = [dos for _ in dims[:-2]]
    logf('Initialising relu network with dimensions',
          f'${'~>'.join(str(d) for d in dims)}$ and dropout probabilities',
         f'$*~>{('~>'.join(str(d) for d in dos))}$...')
    ret = CompLin(dims[0], dims[1], rngs = k)
    for d_in, d_out, do in zip(dims[1:], dims[2:], dos):
        if do:
            ret >>= Dropout(do, rngs = k)
        ret >>= CompLin(d_in, d_out, rngs = k) <<act
    return ret


def dc_relu(k: Rngs, d_in: int, d_out: int, d_hidden_init: int,
            n_hidden: int = 3, decay: float = 2,
            logf: Callable = print, dos = None)->Sequential:
    '''Feedforward elu network with exponentially decaying layer width'''
    d_inners = [round(d_hidden_init/(decay**i)) for i in range(n_hidden)]
    dims = [d_in] + d_inners + [d_out]
    return mlp(k, dims, dos, logf = logf)


def emb_to_likelihood(k, d_emb: int, d_qkv: int,
                      logf:Callable = print)->Sequential:
    logf('Initialising embedding->likelihood mapping with input dimension',
         d_emb, 'and qkv dimension', d_qkv, '...')
    return CompLin(d_emb, d_qkv, rngs = k) >>\
           tanh >>\
           CompLin(d_qkv, 1, rngs = k)


def entuple(x: Any)->tuple[Any]:
    return (x,)


def detuple(x: tuple[Any, ...])->Any:
    return x[0]


def group_tups(lens_in: Iterable[int], lens_out: Iterable[int],
               logf: Callable = print)->Callable[..., Any]:
    l_in, l_out = array(lens_in), array(lens_out)
    logf(f'Setting grouping with signature: ${l_in}~>{l_out}$...')
    n_args = l_in.sum()
    assert n_args == sum(l_out), "In and out indices don't match!"

    tups_inds_in = [(tup, ind) for (tup, n) in\
                    enumerate(l_in) for ind in range(n)]

    breaks = concat([array([0]), l_out.cumsum()])

    map_tab = tuple([tuple(tups_inds_in[s:e]) for (s,e) in\
                    zip(breaks[:-1], breaks[1:])])

    pre = Parallel(*[(entuple if n==1 else Id) for n in l_in])
    post = Parallel(*[(detuple if n==1 else Id) for n in l_out])

    @Fun
    def group(*a):
        ret = pre(*a)
        ret = tuple([tuple([ret[ti][ii] for (ti, ii) in m]) for m in map_tab])
        ret = post(*ret)
        return ret

    return group


@Fun
def nan_inds_to_neg_inf(x: Array)->Array:
    return -inf * (isnan(x.take(indices = array([0]), axis = -1)))


@Fun
def add(a:Array, b:Array)->Array:
    return a + b


def exp_emb_ill(k: Rngs, d_in: int, d_emb: int = 16, n_hidden_emb: int = 3,
                d_qkv: int = 8, d_end_init: int = 32, n_hidden_end: int = 3,
                decay: float = 2, logf: Callable = print,d_emb_init = None,
                dos:float|None = None, decay_end: float|None = None)->Sequential:
    logf('Generating instance level learner...')

    layer_norm = LayerNorm(d_in, rngs = k)
    if d_emb_init is None:
        d_emb_init = round(d_emb*(decay**n_hidden_emb))
    embedding = dc_relu(k, d_in = d_in, d_out = d_emb,
                        d_hidden_init = d_emb_init, n_hidden = n_hidden_emb,
                        decay = decay, logf = logf, dos = dos)

    likelihood = emb_to_likelihood(k, d_emb = d_emb,
                                   d_qkv = d_qkv, logf = logf)
    gt = group_tups((1, 2), (2, 1), logf = logf)

    if decay_end is None:
        decay_end = decay
    final = dc_relu(k, d_in = d_emb, d_out = 1, d_hidden_init = d_end_init,
                    n_hidden = n_hidden_end, decay = decay_end,
                    logf = logf, dos = dos)

    return (nan_inds_to_neg_inf &\
            (nan_to_0 >> layer_norm >> embedding >> (likelihood & Id))) >>\
           gt >>\
           ((add >> sm_bag) | Id) >>\
           expected_embedding >>\
           final >> squeeze


def mha_ill(k: Rngs, d_in: int, d_emb: int = 16, n_hidden_emb: int = 3,
            d_qkv: int = 8, d_end_init: int = 32, n_hidden_end: int = 3,
            decay: int = 2, logf: Callable = print,
            dos:float|None = None)->Sequential:
    logf('Generating instance level learner...')

    d_emb_h = round(d_emb*(decay**n_hidden_emb))
    embedding = dc_relu(k, d_in, d_emb, d_emb_h, n_hidden = n_hidden_emb,
                        decay = decay, logf = logf, dos = dos)

    likelihood = emb_to_likelihood(k, d_emb = d_emb,
                                   d_qkv = d_qkv, logf = logf)
    gt = group_tups((1, 2), (2, 1))

    final = dc_relu(k, d_emb, 1, d_end_init, n_hidden = n_hidden_end,
                    decay = decay, logf = logf, dos = dos)

    return (nan_inds_to_neg_inf &\
            (nan_to_0 >> embedding >> (likelihood & Id))) >>\
           gt >>\
           ((add >> sm_bag) | Id) >>\
           expected_embedding >>\
           final >> squeeze


architectures = {'exp_emb_ill':exp_emb_ill, 'mlp':mlp,
                 'dc_relu':dc_relu, 'mha_ill':mha_ill}
