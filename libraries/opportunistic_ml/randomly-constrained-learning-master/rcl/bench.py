'''
Benchmarking
'''
from jax import Array
from jax.numpy import argmin, flip, array, nonzero
from jax.tree import map as mp

from flax.nnx import Pytree, vmap


def ordered_preds_and_errs(preds: Array, targs: Array)->dict[str, Array]:
    preds = preds.squeeze()
    inds = preds.argsort()
    sprds = preds[inds]
    strgs = array(targs).squeeze()[inds]
    fns = strgs.cumsum() # could be smarter in getting fps but let's be explicit
    fps = flip((flip(1-strgs)).cumsum())
    fps_fns = {'fps':fps, 'fns':fns}

    n = len(targs)
    fp_fn = {'fp':fps/n, 'fn':fns/n}

    np = targs.sum()
    fpr_fnr = {'fpr':fps/(n - np), 'fnr':fns/np}

    return dict(pred = sprds, **fp_fn, **fpr_fnr, **fps_fns)


@vmap(in_axes = (0, None), out_axes = 0) # epochs
def opreds_errs(preds, targs):
    return mp(ordered_preds_and_errs, preds, targs)


@vmap(in_axes = (None, 0), out_axes = 0) #cost ratio
@vmap(in_axes = (0, None), out_axes = 0) #epochs
def opreds_costs(oprs_errs, cr):
  for v in oprs_errs.values():
    v['cost'] = v['fp'] + cr * v['fn']
  return oprs_errs


def select_ind(tr, j):
    return mp(lambda a: a[j], tr)


def best_cutoff(costs_errs: Pytree)->Pytree:
    i = argmin(costs_errs['cost'])
    return select_ind(costs_errs, i)


def first_occurance(arr):
    return nonzero(arr, size = 1)


@vmap(in_axes = (0, None), out_axes = 0) # cost ratio
@vmap(in_axes = (0, None), out_axes = 0) # epochs
def cost_at_cutoff(preds_costs, cutoff_by:str = 'train')->Pytree:
    fp_fn_cutoff_by = best_cutoff(preds_costs.pop(cutoff_by))
    co = fp_fn_cutoff_by['pred']

    inds = {k:first_occurance(v['pred']>=co) for k, v in preds_costs.items()}
    preds_costs = {k:{c:v[i] for c, v in preds_costs[k].items()} for\
                   k, i in inds.items()}
    preds_costs[cutoff_by] = fp_fn_cutoff_by
    return preds_costs


@vmap(in_axes = (0, None), out_axes = 0) # cost ratio
def best_performance(costs_by_epoch, rank_by:str = 'train'):
    c = costs_by_epoch[rank_by]['cost']
    best_ind = argmin(c)
    best_ep = best_ind // c.shape[1]
    #best_cutoff_ind = best_ind // c.shape[0]
    #best_cutoff = costs_by_epoch
    costs_best_epoch = select_ind(costs_by_epoch, best_ep)
    costs = {f'{k}_{c}':v for c, d in costs_best_epoch.items() for k, v in d.items()}
    costs['epoch'] = best_ep
    return costs


