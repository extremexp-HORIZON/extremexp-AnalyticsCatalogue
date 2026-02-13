"""
Core functionality for randomly constrained learning.
"""
__docformat__ = 'numpy'
from itertools import product
from datetime import datetime
from typing import Sequence, Callable
from os import PathLike

from numpy import ndarray, log2, isfinite,\
                           geomspace
from numpy.typing import DTypeLike

from jax.numpy import array
from jax.random import split, key, bits


from optax import adam
from optax.losses import sigmoid_binary_cross_entropy


from pathlib import Path

from rs import Resampler

from rcl.util import tag_logger
from rcl.flaxut import Outcome, Results, NNState, UpdateRule,\
                       UpdateRuleImplementation, TrainingRule,\
                       TrainingCheckpoint
from rcl.flaxar import NN

if  hasattr(__builtins__,'__IPYTHON__'):

    from IPython.display import clear_output as _cl
    from IPython.display import display, Markdown
    def _print(*a, end = '\n', sep = ' '):
        if end != '\n':
            print(*a, end = end, sep = sep)
        else:
            display(Markdown(sep.join([str(aa) for aa in a]) + end))

else:

    _print = print

    def _cl():
        pass


class NNPL:
    """
    Pipeline for training and benchmarking NNs
    """
    p: float
    """Train dataset positive class proportion"""
    p_val: float
    """Val dataset positive class proportion"""
    ds_name: str
    """Name of dataset considered"""
    rs_dir: Path
    """Directory in which to cache resampled data"""
    leaderboard_file: Path
    """Directory in which to store experimental outcomes"""
    resampler: Resampler
    """Resampler class instance"""
    update_rules: set[UpdateRule]
    """Update rules known to the instance"""
    untrained: set[TrainingRule]
    """Not-yet-applied training rules"""
    trained: set[TrainingRule]
    """Completed training rules"""
    nan: set[TrainingCheckpoint]
    """Training outcomes with nans"""
    res: dict[TrainingCheckpoint,Results]
    """Benchmarking results"""
    cost_rats: Sequence[float]
    """Cost ratios (cost(fn)/cost(fp)) of interest"""

    def __init__(self, x_train: ndarray, y_train: ndarray,
                 x_val: ndarray, y_val: ndarray,
                 x_test: ndarray, y_test: ndarray,
                 ds_name: str,
                 cost_rats: tuple[float, ...] = list(geomspace(1,8,7)),
                 loss: Callable=sigmoid_binary_cross_entropy,
                 x_dt:DTypeLike=None, y_dt: DTypeLike=None,
                 rs: Resampler|str|Sequence[str] = Resampler.available(),
                 rs_dir: str|PathLike|None = 'resampled',
                 oc_dir: str|PathLike|None = 'out',
                 n_epochs: int|Sequence[int] = 128, bs: int = 256,
                 lr: float = 1e-4,
                 loss_param: None|tuple|Sequence[None|tuple] = [None],
                 arch: tuple[int, ...] = (128, 64, 32, 1),
                 seed: int = 1729, log: Callable = _print, cl: Callable=_cl,
                 same_seed: bool = True):
        r"""
        Instantiate a new pipeline for NN training.

        Parameters
        ----------
        x_train: ndarray
            Training features
        y_train: ndarray
            Training targets
        x_val: ndarray
            Val features
        y_val: ndarray
            Val targets
        ds_name: str
            Name of the dataset - used for resampled data caching
        cost_rats: tuple[float, ...] = (1,3,10,30,100,300,1000,3000)
            Tuple of cost ratios to use for benchmarking
        loss: Callable = sigmoid_binary_cross_entropy
            Parametric loss family of interest
        x_dt: None|DTypeLike = None
            Datatype for features
        y_dt: None|DTypeLike = None
            Datatype for targets
        rs: Resampler|str|list[str,...] =Resampler.available()
            Resampling schemes to apply.
            - If a Resampler class is passed, infer schemes from that
            - otherwise initialise a new Resampler instance from the provided
            boolean or string(s).
        n_epochs: int|Sequence[int] = 128
            Epochs on which to save snapshots.
            If n_epochs is an int take snapshots
            whenever the epoch number is both
            - a power of two and
            - less than n_epochs
        bs : int|Sequence[int]
            Batch sizes
        lr : float|Sequence[float]
            Learning rates
        loss_param : tuple|None
            Parametric loss arguments
        arch: tuple[int, ...] = (128, 64, 32, 1),
            Shape of feedforward NN to train
        seed: int
            Random state seed
        log: Callable
            Logging callback,
        cl: Callable
            Function call to clear current output cell
        same_seed: bool = True
            If true use the same seed for each NN trained.

        Returns
        -------
        nnpl: NNPL
            Pipeline instance
        """
        self._log = tag_logger(log,'NNPL('+ds_name+')')
        self._log_clean = log
        self._cl=cl

        self._x_train = array(x_train, dtype=x_dt)
        if y_dt is None:
            y_dt = bool  # self._x_train.dtype
        self._y_train = array(y_train, dtype=y_dt)
        self._x_val = array(x_val, dtype=x_dt)
        self._y_val = array(y_val, dtype=y_dt)
        self._x_test = array(x_test, dtype=x_dt)
        self._y_test = array(y_test, dtype=y_dt)

        self._ky = key(seed)
        self.same_seed = same_seed
        if same_seed:
            self.keys = {}

        self.ds_name = ds_name
        d = datetime.now()
        lb = Path('res')/ds_name/(str(d.year)+\
             str(d.month).rjust(2, '0')+\
             str(d.day).rjust(2, '0')+':'+\
             str(d.hour).rjust(2, '0')+ ':'+str(d.minute))
        lb.mkdir(exist_ok = True, parents = True)
        self.leaderboard_file = lb/'leaderboard.txt'
        self.rs_dir = rs_dir
        if isinstance(rs, Resampler):
            self.resampler = rs
            resampling_schemes = self.resampler.schemes
        else:
            self.resampler = Resampler(self._x_train, self._y_train,
                                       self.rs_dir,self.ds_name,
                                       int(bits(self._getk('rs'))))
            resampling_schemes = rs

        self.p = self._y_train.mean()
        self.p_val = self._y_val.mean()

        if arch[-1] != 1:
            arch = arch + (1,)
        self._nn = NN(features=arch)

        self.set_parametric_loss(loss)
        self.update_rules = set()
        self.untrained = set()
        self.trained = set()
        self.nan = set()
        self._nns = {}
        self.res = {}
        self.add_training_rules(lr, bs, loss_param,
                                resampling_schemes, n_epochs)
        self.cost_rats = cost_rats


    def _set_bs(self,bs):
        if bs is not None:
            self.bs = [bs] if isinstance(bs, int) else bs
        assert self.bs, "Batch size not set!"


    def _set_rs(self,rs:str|Sequence[str]|bool):
        if isinstance(rs,bool):
            rs = Resampler.available() if rs else ['']
        elif isinstance(rs,str):
            rs = [rs]

        if isinstance(rs,Sequence):
            self.rs = rs
        elif rs is None:
            assert self.rs, "Resampling schemes not set!"
        else:
            raise TypeError('Unknown resampling schemes '+str(rs))

    def _set_loss_param(self,loss_param:tuple|Sequence[tuple]|None):
        if isinstance(loss_param,tuple):
            self.loss_param = [loss_param]

        if isinstance(loss_param, Sequence):
            self.loss_param = loss_param
        elif loss_param is None:
            assert self.loss_param, "Loss param not set!"
        else:
            raise TypeError('Unknown loss params '+str(loss_param))

    def _set_n_epochs(self,n_ep: int|Sequence[int]|None):
        if isinstance(n_ep,int):
            n_ep = tuple(2**i for i in range(int(1 + log2(n_ep))))

        if isinstance(n_ep, Sequence):
            self.n_epochs = n_ep
        elif n_ep is None:
            assert self.n_epochs, "Epoch checkpoints not set!"
        else:
            raise TypeError('Unknown epoch specification '+str(n_ep))

    def _set_lr(self,lr:float|Sequence[float]|None):
        if isinstance(lr,float):
            lr = [lr]

        if isinstance(lr,Sequence):
            self.lr = lr
        elif lr is None:
            assert self.lr, "Learning rate not set!"
        else:
            raise TypeError('Unknown lr specification '+str(lr))

    def add_training_rules(self, lr:None|bool|float|Sequence[float]=None,
                           bs:None|int|Sequence[int]=None,
                           loss_param:object=None,
                           resampling_schemes:None|bool|str|Sequence[str]=None,
                           n_epochs:None|int|Sequence[int]=None)->None:
        r"""
        Define new training rules on which to train.

        Parameters
        ==========
        n_epochs: int|Sequence[int]
            Epochs on which to save snapshots.
            If n_epochs is an int take snapshots
            whenever the epoch number is both
            - a power of two and
            - less than n_epochs
        bs : int|Sequence[int]
            Batch sizes
        lr : float|Sequence[float]
            Learning rates
        resampling_schemes : str|Sequence[str]
            Strings describing resampling schemes to apply
        loss_param : tuple|None
            Parametric loss arguments
        """
        self._set_bs(bs)
        self._set_rs(resampling_schemes)
        self._set_loss_param(loss_param)
        self._set_n_epochs(n_epochs)
        self._set_lr(lr)
        new_update_rules = {UpdateRule(lr, bs, p) for lr, bs, p in\
                            product(self.lr, self.bs, self.loss_param)}
        new_training_rules = {TrainingRule(self.n_epochs, ur, rs) for\
                              ur, rs in product(new_update_rules, self.rs)}
        new_training_rules -= self.trained
        self.update_rules.update(new_update_rules)
        self.untrained.update(new_training_rules)


    def set_parametric_loss(self, loss):
        """
        Set the parametric loss for the pipeline

        Parameters
        ----------
        loss : Callable[tuple[ndarray[float], ndarray, object],
                        ndarray[float]]
            Parametric function on which to perform gradient descent,
            where loss(preds,targs,par) is the loss value as a function of
            - continuous predictions preds:ndarray[float]
            - target y values targs:ndarray
            - loss function parameter par:object
        """
        self.loss = loss
        self.uri = UpdateRuleImplementation(loss=self.loss,log=self._log,
                                            forward=self._nn.apply)


    def _make_nns(self, lr):
        """Initialise a NN state"""
        param = self._nn.init(self._getk('nn'), self._x_train[0])
        return NNState(0, adam(learning_rate=lr).init(param), param)

    def train(self, tr_rule, store_trained):
        """Train according to a fixed TrainingRule"""
        if tr_rule in self.trained:
            self._log('Already trained', tr_rule, '?!')
            return
        self._nns.update(tr_rule._train(self.uri, self._make_nns(tr_rule.ur.lr),
                                        self.resampler, self._getk('train')))
        if not store_trained:
            self._log('Removing rule...')
            self.uri.rules.pop(tr_rule.ur)
        self._update_res(store_trained)
        self.trained.add(tr_rule)
        self.untrained -= {tr_rule}

    def train_all(self, store_trained=True):
        """Train all rules"""
        [self._log(r) for r in self.untrained]
        for i, tr in enumerate(list(self.untrained)):
            self.leaderboard(save = True)
            if not i%10:
                self._log('Training with rule:', tr,
                         '    (#trained:' + str(len(self.trained)),
                 '#untrained:' + str(len(self.untrained)) + ')')
                self._log('Training according to', len(self.untrained),
                         'distinct rules...')
                self._cl() # clear output of a cell between training rules
                self.leaderboard(topk=2, precision = 2)
            self.train(tr, store_trained)
        self.store_res()

    def _update_res(self, store_trained):
        """Benchmark trained models"""
        newly_trained = {c:self._nns[c] for c in self._nns if\
                         not(c in self.res or c in self.nan)}
        for checkpoint, nns in newly_trained.items():
            self._log('NEWLY TRAINED')
            self._log(checkpoint)
            pred_train= self._nn.apply(nns.param, self._x_train)
            pred_val = self._nn.apply(nns.param, self._x_val)
            pred_test = self._nn.apply(nns.param, self._x_test)
            if all(isfinite(pred_train)):
                self.res[checkpoint] = Results.\
                                       from_predictions(pred_train,
                                                        self._y_train,
                                                        pred_val,
                                                        self._y_val,
                                                        pred_test,
                                                        self._y_test,
                                                        self.cost_rats)
            else:
                self.nan.add(checkpoint)
            if not store_trained:
                self._log('Removing trained nn...')
                self._nns.pop(checkpoint)

    def store_res(self)->None:
        """
        Save results to a text file.
        """
        self.leaderboard(save=True)

    def _getk(self, use: str)->key:
        """Get random state for prng purposes"""
        if self.same_seed:
            if use not in self.keys:
                self._ky, self.keys[use] = split(self._ky)
            return self.keys[use]
        else:
            self._ky, k = split(self._ky)
            return k

    def leaderboard(self, topk: int|None = None, by: str = 'val',
                    ls_nans=10, ret: bool = False, save:bool = False,
                    precision: int = 0)->None|str:
        if save:
            self.leaderboard_file.write_text(self.leaderboard(ret = True,
                                                              ls_nans=None))
            return
        """Rank training rules by performance for varying statistics"""
        self._log('LEADERBOARD')
        lg=self._log_clean
        self._r = ''
        if ret:
            def lg(*x):
                self._r += ' '.join([str(t) for t in x]) + '\n'

        lg('$\\mathbb P(+|$train$)=',self._y_train.mean(),'$')
        lg('$\\mathbb P(+|$val$)=',self._y_val.mean(),'$')
        self.res_by_rat(by)
        for cr, res in self._res_all.items():
            lg('### cost ratio:',cr)
            head = '|' + TrainingCheckpoint._h + '|' + Outcome._h + '|'
            l_br = ''
            for i,h in list(enumerate(head)):
                l_br += '|' if h == '|' else '-'
            tab = [head, l_br]
            for r in res[:topk]:
                tab.append('|' + r[1].r + '|' + r[0].r(precision) +'|')
            lg('\n'.join(tab))
        if ret:
            return self._r+'\n'

        if len(self.nan):
            if ls_nans:
                lg('Training that resulted in nans:')
                [lg(r) for r in list(self.nan)[:ls_nans]]
                if isinstance(ls_nans,int) and ls_nans<len(self.nan):
                    lg('...')
            else:
                lg(len(self.nan),'training outcomes have nans')
        else:
            lg('No models have nan issues :)')

    def res_by_rat(self, by: str)->None:
        """Get all results obtained by the pipeline"""
        self._res_all = {r:[] for r in self.cost_rats}
        p_tr = self._y_train.mean()
        n_tr = 1 - p_tr
        p_te = self._y_test.mean()
        n_te = 1 - p_te
        p_val = self._y_val.mean()
        n_val = 1 - p_val
        for checkpoint, res in self.res.items():
            for cost_rat, cost_trn, cost_val, cost_tst, _, e_train, e_val, e_test in res:
                self._res_all[cost_rat].append((Outcome(cost_trn, cost_val,
                                                        cost_tst,
                                                        e_train.fp[0] / n_tr,
                                                        e_train.fn[0] / p_tr,
                                                        e_val.fp[0] / n_val,
                                                        e_val.fn[0] / p_val,
                                                        e_test.fp[0] / n_te,
                                                        e_test.fn[0] / p_te),
                                                checkpoint))
        for res in self._res_all.values():
            if by == 'train':
                res.sort()
            elif by == 'val':
                res.sort(key=lambda x:x[0].cost_val)
            else:
                raise ValueError('Invalid sorting option',by)
        return self._res_all


def benchmark_nnpl(n:NNPL, x: array, y: array)->\
tuple[[float],[float],[float],[TrainingCheckpoint]]:
    """Test trained models"""
    ret = (fprs, fnrs, rats, best) =  ([], [], [], [])
    p_pos_val = y.mean()
    p_neg_val = 1 - p_pos_val
    for cr, outcomes in n.res_by_rat('val').items():
        rats.append(cr)
        best_res, best_checkpoint = outcomes[0]
        best.append(best_checkpoint)
        print('Cost ratio:', cr)
        print('Best validation performance:', best_checkpoint)
        print(best_res.r(2))
        print('Worst validation performance:', outcomes[-1][1])
        print(outcomes[-1][0].r(2))
        res_cp = n.res[best_checkpoint]
        cr_index = res_cp.cost_rats.index(cr)
        thresh = res_cp.cutoffs[cr_index]
        print(len(n._nns),'trained nns available...')
        pred = (n._nn.apply(n._nns[best_checkpoint].param, x) > thresh).\
               reshape((-1,))
        print('P(+)',p_pos_val)
        print('P(predict +)',pred.mean())
        fpr = (pred & ~y).mean() / p_neg_val
        fnr = ((~pred) & y).mean() / p_pos_val
        fprs.append(fpr)
        fnrs.append(fnr)
        print('Test performance for checkpoint with best validation performance:')
        print('fpr:', fpr)
        print('fnr:', fnr)
    print()
    return ret

_notes_header ="""
Notes
-----
"""

lines = NNPL.__init__.__doc__.split('\n')
n_leading_spaces = min([len(ln)-len(ln.lstrip()) for ln in lines if ln])
NNPL.__init__.__doc__ = '\n'.join([ln[n_leading_spaces:] for ln in lines])
NNPL.__init__.__doc__ +=_notes_header +(Path(__file__).parent.parent/'md'/'nnpl.md').read_text()
