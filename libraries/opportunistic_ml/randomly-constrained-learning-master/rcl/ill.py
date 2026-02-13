from warnings import warn
from pathlib import Path
from pickle import dumps, loads
from json import dumps as dumpj, loads as loadj
from typing import Any, TextIO
from collections.abc import Iterable, Sized, Callable
from multiprocessing.connection import  Connection

from numpy import array as nparr, format_float_positional, inf
from numpy.typing import DTypeLike, ArrayLike

from flax.nnx import Rngs, Optimizer, grad, scan, Param, Carry,\
                     graphdef, state, cond, merge, Pytree, update

from optax import adam

from jax import device_put, devices, Array
from jax.lax import map as mlp
from jax.numpy import array, concat, isfinite, argmin, nan, expand_dims, stack
from jax.debug import callback
from jax.random import key, split
from jax.tree import map as mp

from orbax.checkpoint import AsyncCheckpointer, StandardCheckpointHandler
from orbax.checkpoint.args import StandardRestore, StandardSave

from pandas import DataFrame

from dl import rbd24, rbd24_recommended

from rcl.loss import losses
from rcl.util import diter, tag_logger, add_leading_axis, tree_trunc, get_def,\
                     lists_to_tups, cp, ensure_exists, pad_and_stack,\
                     to_hashstr, md_tab, get_varying
from rcl.bench import opreds_errs, opreds_costs,\
                      cost_at_cutoff, best_performance
from olj.arch import architectures


def summarise_il_dfs(dfs: dict[str, DataFrame], instance_lab: str = 'user_id',
                    target_lab: str = 'label', i = 'users')->None:
    rows: list[tuple[int, str]] = []
    u_tab: str = f'|category|#{i}|#benign ({i})|#malicious ({i})|#samples|'
    u_tab += f'#benign (samples)|#malicious (samples)|imbalance ({i})|'
    u_tab += f'imbalance (samples)|labels constant for fixed {i}?|\n'
    u_tab += '|-|-|-|-|-|-|-|-|-|-|\n|'

    for cat, df in dfs.items():
        y = df[target_lab]
        t_range = set(y)
        assert t_range == {0,1}, 'unexpected range for targets:' + str(t_range)
        y = y.astype(bool)
        u = nparr(df[instance_lab])
        samples = (u, u[~y], u[y])
        v, v0, v1 = (len(w) for w in samples)
        ua, u0, u1 = (len(set(v)) for v in samples)
        unique_instance_labels = 'yes' if u0 + u1 == ua else 'no'
        r_s = '|'.join(str(s) for s in\
                         [cat, ua, u0, u1, v, v0, v1,
                         format_float_positional(u1/(u0+u1), precision = 3),
                         format_float_positional(v1/(v0+v1), precision = 3),
                         unique_instance_labels])
        rows.append((ua, r_s))
    rows.sort(reverse = True)
    _, r_str = zip(*rows)

    u_tab += '|\n|'.join(r_str) + '|\n'
    return u_tab


def df_to_bags(instance_rows: Iterable[DTypeLike],
               feature_rows: Iterable[Iterable[float]],
               target_rows: Iterable[float], multi_targ: None|bool = None)\
->tuple[ArrayLike, tuple[ArrayLike, ...],\
        ArrayLike|tuple[ArrayLike, ...], bool]:
    """Convert a collection of rows into instances with associated bags"""

    unique_targ = True
    bag_lab:dict[DTypeLike, tuple[list[Iterable[float]], list[float]]] =\
    {i:([], []) for i in set(instance_rows)}
    for i, f, t in zip(instance_rows, feature_rows, nparr(target_rows)):
        b, m = bag_lab[i]
        b.append(f)
        if t not in m:
            if  len(m):
                unique_targ = False
                if not multi_targ:
                    msg = 'Target for instance ' + str(i) + 'is nonunique!'
                    if multi_targ is None:
                        warn(msg)
                    else:
                        raise AssertionError(msg)
            m.append(t)

    bags: tuple[Array, ...] = ()
    instances, bags,\
    targs_l = zip(*sorted((i, nparr(b), nparr(m)) for i, (b, m) in\
                          bag_lab.items()))
    if unique_targ:
        targets = nparr(targs_l).flatten()

    return instances, bags, targets, unique_targ


default_proportions = {'train':.6, 'val':.2, 'test':.2}


def rbd24_bagged(select_dataset:bool|str|Iterable[str] = rbd24_recommended,
                 proportions: dict[str, float] = default_proportions,
                 seed:int = 1729,
                 n_folds:int = 0)->tuple[dict, dict, dict, dict]:
    if n_folds:
        proportions = {str(i):1/n_folds for i in range(n_folds)}
    features_rows, targets_rows, indices_by_split, dfs,\
    colnames = rbd24(proportions = proportions, random_state = seed,
                     select_dataset = select_dataset)
    selected = list(dfs)
    splits = list(proportions)
    users, bags, targets = ({s:{} for s in selected} for _ in range(3))

    for (c, feats_r_df, targs_r_df, inds_r_df, users_df, bags_df, targets_df,\
         df) in diter(selected, features_rows, targets_rows, indices_by_split,
                      users, bags, targets, dfs):

        us = df['user_id']
        users_r_df = {spl:us[inds] for spl, inds in inds_r_df.items()}

        for (spl, f_rows, t_rows, u_rows) in\
            diter(splits, feats_r_df, targs_r_df, users_r_df):

            users_df[spl], bags_df[spl], targets_df[spl],\
            unique_labs = df_to_bags(u_rows, f_rows, t_rows)

            assert unique_labs, "Expected unique instance labels!!"

    return users, bags, targets, dfs


def summarise_bags(bags: Iterable[Sized], targets = None)->str:
    ret = f'- Number of bags: {len(bags)}\n'
    if targets is not None:
        ret += f'- #positive: {targets.sum()}\n'
    sizes = nparr([len(b) for b in bags])
    ret += f'- Smallest bag size: {sizes.min()}\n'
    ret += f'- Largest bag size: {sizes.max()}\n'
    ret += f'- Mean bag size: {sizes.mean()}\n'
    return ret


class Cache:
    def __init__(self, root:str|Path = '.ill', logger:Callable = print):
        self._root = (Path(root) if isinstance(root, str) else root).resolve()

    @property
    @ensure_exists
    def root_dir(self):
        return self.root


    _hph = None
    @property
    def hp_id(self):
        if self._hph is None:
            self._hph = to_hashstr(self.hp)
        return self._hph


    @hp_id.setter
    def hp_id(self, h):
        self. _hph = h


    @property
    def d_in(self):
        return list(self.data['features'].values())[0][0].shape[-1]

    def get_cached_data(self):
        try:
            return loads(self.dataf.read_bytes())
        except FileNotFoundError:
            return None


    _hp = None
    @property
    def hp(self):
        if self._hp is None: # try to load
            hp = loadj(self.expf.read_text()[:-1])
            self._hp = lists_to_tups(hp)
        return self._hp


    @hp.setter
    def hp(self, v):
        for k in self.optional_hp():
            if k in v and v[k] is None:
                v.pop(k)
        self._hp = v
        if not self.expf.exists():
            self.expf.write_text(dumpj(v) + '\n')


    @property
    @ensure_exists
    def lab_dir(self):
        return self.ill_dir / self.lab


    @property
    def dataf(self):
        return self.lab_dir / 'data.pkl'


    @property
    def datah(self):
        ret = self.lab_dir / 'data.hash'
        if not ret.exists():
            ret.write_text('')
        return ret


    @property
    @ensure_exists
    def lab_seed_dir(self):
        return self.lab_dir / str(self.seed)


    @property
    @ensure_exists
    def res_dir(self):
        return self.lab_seed_dir / self.hp_id


    @property
    @ensure_exists
    def cr_dir(self):
        return self.res_dir / 'cr'


    @property
    def nan_ep_file(self):
        return self.res_dir / 'nan_epoch'


    @property
    def expf(self):
        return self.ill_dir / f'{self.hp_id}.json'


'''
def res_to_tab(res:Pytree, crs:Iterable[float], exclude:None|set[str] = None,
               include:set[str]|None = None, by:str = 'cost_val')->str|dict:
    if exclude is None:
        exclude = set([f'{a}_{b}' for a in ['fp', 'fn', 'fps', 'fns'] for\
                       b in ['train', 'val', 'test']])
    res_keys = sorted(list(res))
    res_keys.sort(key = lambda s: - ('val' in s) - 2 * ('test' in s))
    res_keys = list(filter(lambda a: a not in exclude, res_keys))
    if include:
      res_keys = list(filter(lambda a: a in include, res_keys))
    res_values = list(res[k] for k in res_keys)
    heads = ['cr'] + res_keys
    rows = []
    for i, cr in enumerate(crs):
        r = [cr] + [v[i] for v in res_values]
        rows.append(r)

    return rows
'''

class ILL:
    def __init__(self, lab:str, seed:int = 1729, fold:int|None = None,
                  n_folds:int|None = None,loss_kwargs:Any = {'param':(1.,2.),
                                    'key':'fp_fn_perturbed_bce'},
                 loss_key:None|str = None, data:dict|None = None,
                 cost_rats = tuple(10**(t/10) for t in range(11)),
                 lr:float = 1e-3, restore: str = '',
                 dropout = None, d_emb_init: int|None = None, d_emb: int = 16,
                 n_hidden_emb: int = 3, d_qkv: int = 8,
                 decay_end: float|None = None, d_end_init: int = 32,
                 n_hidden_end: int = 3, decay: int = 2, bs:int = None,
                 logger:Callable = print, log_out: None|TextIO = None):
        self.log_out =  Path(log_out) if isinstance(log_out, str) else log_out
        self.seed = seed
        self.lab = lab
        self.fold = fold
        self.n_folds = n_folds
        if self.n_folds:
            self.lab += f'{n_folds}_folds'
            if data:
                assert set(range(self.n_folds)) == {int(k) for k in\
                                                    data['features']}
        self._logger = logger
        self._l = tag_logger(logger, f'{seed}:{lab}', self.log_out)
        self.cost_rats = cost_rats

        if loss_key is None:
            loss_key = loss_kwargs['key']

        if data:
            self.data = data

        if restore:
            self._l('Initialising', type(self).__name__, 'instance with seed',
                    seed, 'from loaded parameters: {restore}...')
            self.hp_id = restore

        else:
            lkwa = dict(**loss_kwargs)
            lkwa.pop('key', None)
            self._l('Initialising', type(self).__name__, 'instance with seed',
                    seed, 'from passed parameters...')
            hp = {'d_emb':d_emb, 'n_hidden_emb':n_hidden_emb, 'd_qkv':d_qkv,
                  'd_end_init':d_end_init,'seed':seed,
                  'n_hidden_end':n_hidden_end, 'decay':decay, 'lr':lr,
                  'loss_kwargs':lkwa, 'loss_key':loss_key,
                  'dropout':dropout, 'd_emb_init':d_emb_init,
                  'decay_end':decay_end, 'bs':bs}
            self.hp = hp
        self._l = tag_logger(logger, f'{self.hp_id}:{seed}:{lab}',
                              self.log_out)

        self._init_model()
        self.restore_state_and_res()

        @grad #Note this is the nnx version
        def dl(imp, feat, targ):
            return self.loss(imp(feat), targ).sum()
        self.dl = dl

        @scan(in_axes = (Carry, 0, 0), out_axes = Carry)
        def steps(c, feat, targ):
            mod, opt = c
            mod.train()
            g =dl(mod, feat, targ) #self.dl(mod, feat, targ)
            #gf = flatten(mp(lambda a: (a**2).sum(), g))
            opt.update(mod, g)
            return c
        self.steps = steps


    @property
    @ensure_exists
    def ill_dir(self):
        return Path('./.ill').resolve()


    _hph = None
    @property
    def hp_id(self):
        if self._hph is None:
            self._hph = to_hashstr(self.hp)
        return self._hph


    @hp_id.setter
    def hp_id(self, h):
        self. _hph = h


    def res_tab(self, crs:Iterable[float]|None = None, include = None,
                exclude:set[str]|None = None, cutoffs_by:str|None = None,
                sort_by:str|None = 'train')->str|dict:

        if crs is None:
            crs = self.cost_rats
        res = self.results(crs, cutoffs_by = cutoffs_by)
        return res_to_tab(res, crs, exclude, by = sort_by, include = include)


    res = None
    def results(self, crs:Iterable[float], to_pkl = False, to_csv = False,
                cutoffs_by:str|None = 'train', rank_by:str = 'train')->Pytree:
        crs = array(crs)
        d = self.tr_val_features_targets() #self.data['targets']
        t = d['targets']

        by_dir = self.cr_dir / cutoffs_by
        by_dir.mkdir(exist_ok = True)
        res_file = by_dir / 'res.pkl'
        meta_file = by_dir / 'res.json'
        res = None
        res_meta = cr_ep_json(crs, self.n_epochs, cutoffs_by)
        if res is not None and self.res_meta == res_meta:
            res = self.res
        elif meta_file.exists() and res_meta == meta_file.read_text():
            res = loads(res_file.read_bytes())
            to_pkl = False

        else:

            yp = self.yp
            self.oerrs = opreds_errs(yp, t)
            costs_all_epochs = opreds_costs(self.oerrs, crs)

            res_all_ep = self.res_all_ep = cost_at_cutoff(costs_all_epochs,
                                                          cutoffs_by)
            res = self.res = best_performance(res_all_ep)
            self._l(list(res))

        self.res_meta = res_meta
        if to_pkl:
            self._l(f'Saving to {res_file}... ', end = '')
            res_file.write_bytes(dumps(res))
            self._l(f'crs and n_epochs to {res_file}... ', end = '')
            meta_file.write_text(res_meta)
        '''
        if to_csv:
            res_r = {k:nparr(v) for k, v in res.items()}
            resk = sorted(res)
            for i, cr in enumerate(crs):
                rows = [['epoch'] + resk]
                for ep in range(self.res_len):
                    rows.append([str(ep)] + [str(res_r[k][i]) for\
                                             k in resk])
                self._l()
                rows = [f'{','.join(r)}' for r in rows]
                cr_file = by_dir / f'{cr}.csv'
                cr_file.write_text('\n'.join(rows) + '\n')
                self._l(f'Results saved to {cr_file}!', end = '')
        '''
        self._l()
        return res


    @property
    def d_in(self):
        return list(self.data['features'].values())[0][0].shape[-1]

    def get_cached_data(self):
        try:
            return loads(self.dataf.read_bytes())
        except FileNotFoundError:
            return None


    _data = None
    @property
    def data(self)->None|dict[str, dict[str, Array]]:
        if self._data is None:
            self._data = self.get_cached_data()
        return self._data


    @data.setter
    def data(self, d:None|dict[str, dict[str, Array]]):
        '''
        if any((d is not None) for d in \
                (feats_train, targs_train, feats_val, targs_val,\
                 feats_test, targs_test)):
            self.data = {'features':{'train':pad_and_stack(feats_train),
                                     'val':pad_and_stack(feats_val),
                                     'test': (None if feats_test is None else\
                                              pad_and_stack(feats_test))},
                         'targets':{'train':targs_train, 'val':targs_val,
                                    'test':targs_test}}
        '''
        d_str = dumps(d)
        d_hash = to_hashstr(d_str)
        cached = self.datah.read_text()
        if not cached:
            self._l(f'Cacheing data with hash {d_hash} in {self.dataf}...')
            self.dataf.write_bytes(d_str)
            self.datah.write_text(d_hash)
            self._data = d
        else:
            assert d_hash == cached, "Passed and cached data don't match"
            self._l('Note: identical data to that passed to ILL()',
                    f'has already been cached at {self.dataf}..')
        self._data = d


    _hp = None
    @property
    def hp(self):
        if self._hp is None: # try to load
            hp = loadj(self.expf.read_text()[:-1])
            self._hp = lists_to_tups(hp)
        return self._hp


    @hp.setter
    def hp(self, v):
        for k in self.optional_hp():
            if k in v and v[k] is None:
                v.pop(k)
        self._hp = v
        if not self.expf.exists():
            self.expf.write_text(dumpj(v) + '\n')


    def __getitem__(self, k):
        if k in self.hp:
            return self.hp[k]
        elif k in self.optional_hp():
            return None
        else:
            raise KeyError(f'Undefined hyperparameter: {k}')


    def _init_model(self):
        self._l('Initialising rngs...')
        kp, ks, kd = split(key(self.seed), 3)
        self.rngs = Rngs(params = kp, shuffle_train = ks, dropout = kd)

        self._l('Initialising model...')
        arch_key = 'exp_emb' if self['arch'] is None else self['arch']
        arch =architectures[arch_key + '_ill']
        self.mod = arch(self.rngs, self.d_in, d_emb = self['d_emb'],
                        n_hidden_emb = self['n_hidden_emb'],
                        d_qkv = self['d_qkv'], d_end_init = self['d_end_init'],
                        n_hidden_end = self['n_hidden_end'],
                        decay = self['decay'], logf = self._l,
                        d_emb_init = self['d_emb_init'],
                        decay_end = self['decay_end'],dos = self['dropout'])

        self.cp = AsyncCheckpointer(StandardCheckpointHandler())
        self.opt = Optimizer(self.mod, adam(learning_rate = self['lr']),
                             wrt = Param)

    def loss(self, pred, targ):
        return losses[self['loss_key']](pred, targ, **self['loss_kwargs'])


    history = {0:None}
    def checkpoint_state(self):
        self.history[self.n_epochs] = self.state
        self.save_state()


    yp = None
    @property
    def res_len(self):
        return 0 if self.yp is None else len(self.yp['train'])


    @property
    def n_epochs_unsaved(self):
        ret = self.next_time - self.res_len
        return ret


    def all_epochs_saved(self):
        self.next_time = self.res_len
        self.n_new = None


    @property
    def n_epochs(self):
        return self.next_time - 1


    def increment_epoch(self):
        self.next_time += 1


    _tft = None
    def tr_val_features_targets(self):
        if self.fold is None:
            return self.data
        else:
            if self._tft is None:
                folds = sorted(self.data['features'])
                sf = str(self.fold)
                trf = pad_and_stack(sum((self.data['features'][k] for\
                                         k in folds if k != sf), ()))
                vf = pad_and_stack(self.data['features'][sf])
                trt = concat([self.data['targets'][k] for\
                              k in folds if k != sf])
                vt = self.data['targets'][sf]
                self._tft = {'features':{'train':trf, 'val':vf},
                             'targets':{'train':trt, 'val':vt}}
            return self._tft


    def epochs(self, n_epochs, until = True):
        if self.nan_epoch:
            self._l('Skipping training (nans will be encountered',
                    f'during epoch {self.nan_epoch})...')
            return
        if until:
            n_epochs -= self.n_epochs
            n_epochs = max(0, n_epochs)
        rng = self.rngs.shuffle_train
        d = self.tr_val_features_targets()
        b = {k:array(v) for k, v in d['features'].items()}
        t = {k:array(v) for k, v in d['targets'].items()}
        n_train_targs = len(t['train'])
        xt = b['train']
        yt = t['train']


        bs = self['bs'] if self['bs'] else 1
        n_batches = n_train_targs // bs
        n_bags_in_epoch = n_batches * bs
        self._l(f'Batches per epoch: {n_batches}')
        def still_finite_body(mod, opt, r):
            perm = r.permutation(n_train_targs)
            x, y = xt[perm], yt[perm]
            x = x[:n_bags_in_epoch].reshape(n_batches, bs, -1, self.d_in)
            y = y[:n_bags_in_epoch].reshape(n_batches, bs)

            mod, opt = self.steps((mod, opt), x, y)
            #mod, opt =  self.steps((mod, opt), x, y)
            modev = mp(lambda x:x, mod)
            modev.eval()
            yp = {k:modev(v) for k, v in  b.items()} #get_preds(mod)
            still_finite = isfinite(yp['val'].sum())
            return (mod, opt, r, still_finite), yp

        nan_yp = mp(lambda a: a + nan, t)

        def not_finite_body(mod, opt, r):
            return (mod, opt, r, False), nan_yp

        @scan(in_axes = Carry, out_axes =(Carry, 0), length = n_epochs)
        def scan_eps(c):
            mod, opt, r, still_finite = c
            c, yp = cond(still_finite, still_finite_body,
                         not_finite_body, mod, opt, r)
            _, _, _, still_finite = c
            callback(self.log_end_epoch, still_finite)
            return c, yp

        self._l('Running for', n_epochs, 'epochs... ')
        self._l('(previously completed', self.n_epochs, 'epochs)...') if\
        self.n_epochs else self._l('(no previous training)...')
        self._l(f'(#optimiser steps:{self.opt.step.value})')
        (self.mod, self.opt, self.rngs.shuffle_train, still_finite),\
        yp = scan_eps((self.mod, self.opt, rng, True))
        self.n_new = self.n_epochs_unsaved
        self._l()
        self._l(f'Completed run of {self.n_new} epochs...')
        self._l(f'(#optimiser steps:{self.opt.step.value})')
        if self.n_new:
            self._l(f'saving {self.n_new} results and checkpointing state...')
            if self.nan_epoch is not None:
                yp = {k:tree_trunc(v, self.n_new) for k, v in yp.items()}
            self.set_res(yp)
            self.checkpoint_state()
            if self.cost_rats:
                self._l('Saving according to cost ratios:',
                        *(f'{c:.2f},' for c in self.cost_rats))
                self.results(self.cost_rats, to_pkl = True, to_csv = True,
                             cutoffs_by = 'train')
                self.results(self.cost_rats, to_pkl = True, to_csv = True,
                             cutoffs_by = 'val')
        print('At start:')
        print(*((k, v.shape, v[0].min(), v[0].max(),
                 v[0].mean()) for k, v in self.yp.items()))
        print('By end:')
        print(*((k, v.shape, v[-1].min(), v[-1].max(),
                 v[-1].mean()) for k, v in self.yp.items()))


    def load_res(self):
        assert not self.res_len, 'Results already loaded!'
        ivs = {}
        for i in [j for j in self.res_dir.iterdir() if j.suffix == '.pkl']:
            start_str, end_str = i.stem.split(':')
            ivs[int(start_str), int(end_str)] = loads(i.read_bytes())
        if not ivs:
            d = self.tr_val_features_targets() #self.data
            ft = d['features']
            yp = {}
            for k, v in ft.items():
                v = device_put(v, devices(backend = 'cpu')[0])
                self.mod.eval()
                yp[k] = add_leading_axis(self.mod(v))
                self.mod.train()
            self.write_res(yp)
        else:
            sivs = sorted(ivs.items())
            s = tuple(k[0][0] for k in sivs)
            e = tuple(k[0][1] for k in sivs)
            assert (s[1:] == e[:-1]),\
                   f'Inconsistent intervals in {self.res_dir}!'
            yp = {}
            counts = {}
            for (s, e), z in sivs:
                for k, v in z.items():
                    if k in yp:
                        yp[k].append(v)
                        counts[k] += 1
                    else:
                        yp[k] = [v]
                        counts[k] = 1
                assert len(set(counts.values())) == 1,\
                       'Inconsistent hisory lengths!'
            yp = {k:concat(v) for k, v in yp.items()}
        self.set_res(yp, save = False)


    def write_res(self, yp):

        start = self.res_len
        end = start + len(yp['train'])
        fn = f'{start}:{end}.pkl'

        (self.res_dir / fn).write_bytes(dumps(yp))


    def set_res(self, yp, save = True):
        if save:
            self.write_res(yp)
        if self.yp is None:
            self.yp = yp
        else:
            for k in yp:
                self.yp[k] = concat([self.yp[k], yp[k]])
        self.all_epochs_saved()


    @property
    def gd(self):
        return graphdef(self.mod)


    @property
    def testing(self):
        return bool('test' in self.data['features'])


    @property
    @ensure_exists
    def lab_dir(self):
        return self.ill_dir / self.lab


    @property
    def dataf(self):
        return self.lab_dir / 'data.pkl'


    @property
    def datah(self):
        ret = self.lab_dir / 'data.hash'
        if not ret.exists():
            ret.write_text('')
        return ret


    @property
    @ensure_exists
    def lab_seed_dir(self):
        ret = self.lab_dir / str(self.seed)
        if self.n_folds:
            ret /= f'fold_{self.fold}'
        return ret


    @property
    @ensure_exists
    def res_dir(self):
        return self.lab_seed_dir / self.hp_id


    @property
    @ensure_exists
    def cr_dir(self):
        return self.res_dir / 'cr'


    @property
    def nan_ep_file(self):
        return self.res_dir / 'nan_epoch'


    @property
    def expf(self):
        return self.ill_dir / f'{self.hp_id}.json'


    @property
    def par(self):
        return state(self.mod)


    @property
    def state_no_cp(self):
        return {'opt':self.opt, 'rngs':self.rngs, 'par':self.par}


    @property
    def state_def(self):
        return get_def(self.state_no_cp)


    @property
    def state(self):
        return cp(self.state_no_cp)


    @state.setter
    def state(self, state):
        self.mod = merge(self.gd, state['par'])
        self.opt =state['opt']
        self.rngs = state['rngs']


    def hyp_tab(self, min_width = 20):
        h, s, v = zip(*[(k, '-', str(p)) for k, p in self.hp_all()])
        hyp_tab = ''
        for i in [h, s, v]:
            hyp_tab += f'|{'|'.join(i)}|\n'
        return hyp_tab

    def hp_all(self):
        hp = list(self.hp.items())
        for k in [k for k in self.optional_hp() if k not in self.hp]:
            hp.append((k, None))
        return sorted(hp)


    @property
    def last_cp_time_and_file(self):
        times_names = [(int(p.name), p) for p in self.res_dir.iterdir() if\
                       p.is_dir() and p.name.isdigit()]
        times_names.sort()
        if not times_names:
            times_names = [(0, None)]
        t, n = times_names[-1]
        return (t, n)


    @property
    def last_cp_time(self):
        return self.last_cp_time_and_file[0]


    @property
    def last_cp_file(self):
        return self.last_cp_time_and_file[1]


    @property
    def stdrs(self):
        return StandardRestore(self.state_def, strict = False)


    @property
    def stds(self):
        return StandardSave(self.state_no_cp)


    def restore_state_and_res(self):
        t, n = self.last_cp_time_and_file
        if n is None:
            self._l('No previous models found...')
        else:
            self._l(f'Load checkpoint: {t}')
            self.state = self.cp.restore(n, args = self.stdrs)
        self.load_res()
        if self.nan_ep_file.exists():
            self.nan_epoch = int(self.nan_ep_file.read_text())
        assert self.last_cp_time == self.n_epochs


    def save_state(self):
        self.cp.save(self.res_dir / str(self.n_epochs), args = self.stds)
        self.cp.wait_until_finished()


    nan_epoch = None
    def log_end_epoch(self, still_finite):
        if still_finite:
            self.increment_epoch()
            self._l(f'End of epoch {self.n_epochs}', end = '\r')
        elif self.nan_epoch is None:
            self.nan_epoch = self.next_time
            self._l(f'nan weights occured during epoch {self.nan_epoch}',
                    ':( bypassing epoch routine in scan... ')
            self.nan_ep_file.write_text(f'{self.nan_epoch}\n')


    @staticmethod
    def optional_hp():
        return {'dropout', 'arch', 'd_emb_init', 'decay_end', 'bs'}


def res_to_tab(res:Pytree, crs:Iterable[float], exclude:None|set[str] = None,
               include:set[str]|None = None, by:str = 'cost_val')->str|dict:
    if exclude is None:
        exclude = set([f'{a}_{b}' for a in ['fp', 'fn', 'fps', 'fns'] for\
                       b in ['train', 'val', 'test']])
    res_keys = sorted(list(res))
    res_keys.sort(key = lambda s: - ('val' in s) - 2 * ('test' in s))
    res_keys = list(filter(lambda a: a not in exclude, res_keys))
    if include:
      res_keys = list(filter(lambda a: a in include, res_keys))
    res_values = list(res[k] for k in res_keys)
    heads = ['cr'] + res_keys
    rows = []
    for i, cr in enumerate(crs):
        r = [cr] + [v[i] for v in res_values]
        rows.append(r)

    return rows, heads


def best_res(res_ls, res_hps, by: str = 'cost_val'):
    ret = []
    for i in range(len(res_ls[0][by])):
        cvs = nparr([res[by][i] for res in res_ls])
        ind = argmin(cvs)
        ret.append([v[i] for v in res_ls[ind]] + res_hps[ind])
    return ret


def run_single(params: dict, n_epochs: int, log_out: Path,
               trans:None|Connection = None, cost_by = None):
    log_out.parent.mkdir(exist_ok = True, parents = True)
    outf = log_out.open('a')
    def logger(*a, **k):
        k['flush'] = True
        k['file'] = outf
        print(*a, **k)
    params['logger'] = logger
    logger('Running single il instance!')
    il = ILL(**params)
    il.epochs(n_epochs)
    if cost_by:
        return il.results(crs = (cost_by,), cutoffs_by = 'train')
    ret = {'id':il.hp_id}
    for s in ['val', 'train']:
        rows, heads = il.res_tab(cutoffs_by = s)
        ret[s] = {'rows':rows, 'heads':heads}
    if trans:
        trans.send(ret)
    return ret


def var_par(lab, x, y, params, n_epochs, crs, seed, logf = print, a = False,
            best = None, precision = 4, vp_dir = '.ill/var_par/'):
    passed = dict(lab = lab, params = params, n_epochs = n_epochs,
                  crs = crs, seed = seed)
    h = to_hashstr(passed)

    logf(f'Evaluating {len(params)} choices of parameters for ill...')
    vp_dir = Path(vp_dir).resolve()
    vp_dir.mkdir(exist_ok = True, parents = True)
    cache = vp_dir / f'{h}.pkl'
    if cache.exists():
        c = loads(cache.read_bytes())
        best = c['best']
        rep = c['rep']
        tab = c['tab']
        start = c['n_done']
    else:
        best = {}
        rep = {}
        tab = {}
        start = 0

    md_out = vp_dir / f'{lab}_{h}.md'
    log_out = vp_dir / f'{lab}_{h}.log'
    x = {f'feats_{k}':v for k, v in x.items()}
    y = {f'targs_{k}':v for k, v in y.items()}
    (vp_dir / f'{h}.json').write_text(dumpj(params) + '\n')
    varying_fields = get_varying(params)
    n_par = len(params)
    for param_ind, par in enumerate(params[start:], start):
        par_all = dict(lab = lab, cost_rats = crs, seed = seed,
                       **par, **x, **y)
        logf(f'Evaluating param {param_ind} of {n_par}...')
        proc_ret = run_single(par_all, n_epochs, log_out)
        logf('...received results!')
        hp_id = proc_ret['id']
        rep = f'# Data: {lab}\n\n'
        for s in ['val', 'train']:
            rows = proc_ret[s]['rows']
            heads = proc_ret[s]['heads']
            cv_ind = heads.index('cost_val')
            if s not in best:
                best[s] = [[inf] * (cv_ind + 1)] * len(crs)
            heads += varying_fields
            heads.append('hp_id')

            for i, r in enumerate(rows):
                if r[cv_ind]<best[s][i][cv_ind]:
                    r = [float(t) for t in r]
                    p = [par[k] for k in varying_fields]
                    best[s][i] = r + p + [hp_id]
            tab[s] = md_tab(heads, best[s], precision = precision)
            rep += f'\n## Cutoff derived from {s}:\n'
            rep += tab[s] + '\n'
            rep += md_tab(heads, best[s], precision = precision,
                          include = {f'{a}_{b}' for a in {'fpr', 'fnr'} for\
                                     b in {'train', 'val', 'test'}}) + '\n'
        md_out.write_text(rep)
        c = {'best':best, 'rep':rep, 'tab':tab, 'n_done':param_ind}
        cache.write_bytes(dumps(c))
        logf(rep)
        logf()
    return c


def vp_k_fold(lab, params, n_epochs, crs, cr_by, seed, data, logf = print, a = False,
              best = None, precision = 3, vp_dir = '.ill/kf/'):
    passed = dict(lab = lab, params = params, n_epochs = n_epochs,
                  crs = crs, seed = seed)
    h = to_hashstr(dumps(passed))
    n_folds = len(data['targets'])

    logf(f'Evaluating {len(params)} choices of parameters for ill...')
    vp_dir = Path(vp_dir).resolve()
    vp_dir.mkdir(exist_ok = True, parents = True)
    cache = vp_dir / f'{h}.pkl'
    varying_fields = get_varying(params)
    if cache.exists():
        ps = loads(cache.read_bytes())
        not_started_params = ps['nys']
        partial_params = ps['partial']
        complete_params = ps['complete']
    else:
        not_started_params = [{'c_avg':0, 'cur_fold':0, 'par':p, 'ind':i,
                               'varying':{k:p[k] for k in varying_fields}} for\
                             i, p in enumerate(params)]
        complete_params = []
        partial_params = []

    md_out = vp_dir / f'{lab}_{h}.md'
    log_out = vp_dir / f'{lab}_{h}.log'
    (vp_dir / f'{h}.json').write_text(dumpj(params) + '\n')
    while len(partial_params) or len(not_started_params):
        if len(not_started_params):
            partial_params.insert(0, not_started_params.pop(0))
        cur_best =partial_params[0]
        par =cur_best['par']
        cur_fold = cur_best['cur_fold']
        par_all = dict(lab = lab, cost_rats = crs, seed = seed,
                       data = data, fold = cur_fold, n_folds = n_folds, **par)
        logf(f'Evaluating fold {cur_fold} of param {cur_best['varying']}...')
        cur_res = run_single(par_all, n_epochs, log_out,
                             cost_by = cr_by)
        cost = cur_res['cost_val'].squeeze()
        cur_best['cur_fold'] += 1
        cur_best['fpr_last'] = cur_res['fpr_val'][0]
        cur_best['fnr_last'] = cur_res['fnr_val'][0]
        cur_best['c_last'] = cost
        cur_best['c_avg'] = (cur_fold * cur_best['c_avg'] + cost) /\
                            cur_best['cur_fold']
        if cur_best['cur_fold'] == n_folds:
            complete_params.append(partial_params.pop(0))
        complete_params.sort(key = lambda a:a['c_avg'])
        partial_params.sort(key = lambda a:a['c_avg'])
        logf('...received results!')
        rep = f'# Partially evaluated: {lab}\n\n'
        rep += md_tab(['fpr_last','fnr_last','c_last','c_avg',
                       'cur_fold'], partial_params[:5]) + '\n'
        if complete_params:
            rep += f'# Completely evaluated: {lab}\n\n'
            rep += md_tab(['fpr_last','fnr_last','c_last','c_avg', 'cur_fold'],
                          complete_params[:5])
        logf(rep)

        md_out.write_text(rep)
        c = {'nys':not_started_params, 'partial':partial_params,
             'complete':complete_params}
        cache.write_bytes(dumps(c))
        logf(rep)
        logf()
    return c


def cr_ep_json(crs, ep, by):
    crs = tuple(float(t) for t in crs)
    return dumpj({'crs':tuple(crs), 'n_epochs':ep, 'by':by}) +'\n'


def epochs(mod, opt, d_in, b, t, rngs, dl, n_epochs = 512,
           bs = 16, logf = print, ret_mod = False, preds_otf = True,
           labs = None, predict_bs = None):
    if labs is None:
        labs = b
    if predict_bs is None:
        predict_bs = bs
    n_train_targs = len(t['train'])
    xt = b['train']
    yt = t['train']
    b = {k:b[k] for k in labs}
    t = {k:t[k] for k in labs}
    if not preds_otf:
        logf('Initial predictions...')
        mod.eval()
        yp_start = mp(mod, b)
        mod.train()

    @scan(in_axes = (Carry, 0, 0), out_axes = Carry)
    def steps(mod_opt, feat, targ):
        mod, opt = mod_opt
        mod.train()
        g =dl(mod, feat, targ)
        opt.update(mod, g)
        return mod, opt


    class log_end_epoch:
        nan_epoch = None
        next_epoch = 1
        def __new__(c, still_finite):
            if still_finite:
                logf(f'\r(End of epoch {c.next_epoch}...)', end = '')
            elif c.nan_epoch is None:
                c.nan_epoch = c.next_epoch
                logf(f'nan weights occured during epoch {c.nan_epoch}',
                        ':( bypassing epoch routine in scan... ')
            c.next_epoch += 1

    n_batches = n_train_targs // bs
    n_bags_in_epoch = n_batches * bs
    logf(f'Batches per epoch: {n_batches}')
    def still_finite_body(mod, opt, rngs):
        perm = rngs.shuffle_train.permutation(n_train_targs)
        x, y = xt[perm], yt[perm]
        x = x[:n_bags_in_epoch].reshape(n_batches, bs, -1, d_in)
        y = y[:n_bags_in_epoch].reshape(n_batches, bs)

        mod.train()
        mod, opt = steps((mod, opt), x, y)
        if preds_otf:
            mod.eval()
            yp = {k:mlp(mod, b[k], batch_size = predict_bs) for k in labs}
            mod.train()
            still_finite = isfinite(yp['val'].sum())
            return (mod, opt, rngs, still_finite), yp
        else:
            still_finite = isfinite(mod(x[0]).sum())
            return (mod, opt, rngs, still_finite)

    nan_yp = mp(lambda a: a + nan, t)

    def not_finite_body(mod, opt, r):
        ret = (mod, opt, r, False)
        return (ret, nan_yp) if preds_otf else ret

    @scan(length = n_epochs, in_axes = Carry,
          out_axes =(Carry, 0) if preds_otf else Carry)
    def scan_eps(c):
        mod, opt, rngs, still_finite = c
        ret = cond(still_finite, still_finite_body,
                     not_finite_body, mod, opt, rngs)

        _, _, _, still_finite = ret[0] if preds_otf else ret
        callback(log_end_epoch, still_finite)
        return ret

    logf('Running for', n_epochs, 'epochs... ')
    logf(f'(#optimiser steps:{opt.step.value})')
    all_eps = scan_eps((mod, opt, rngs, True))
    if preds_otf:
        (_, _, _, still_finite), yp = all_eps
    else:
        (_, _, _, still_finite) = all_eps
    logf()
    logf(f'(#optimiser steps:{opt.step.value})')
    if not preds_otf:
        logf('Final predictions...')
        logf('Set to eval mode...')
        mod.eval()
        yp_end = mp(mod, b)
        yp = mp(lambda a, b:stack([a, b]), yp_start, yp_end)
    logf('At start:')
    logf(*((k, v[0].min(), v[0].max(),
            v[0].mean()) for k, v in yp.items()))
    logf('By end:')
    logf(*((k, v[-1].min(), v[-1].max(),
             v[-1].mean()) for k, v in yp.items()))
    return (yp, (mod, opt, rngs)) if ret_mod else yp
