"""
Core functionality for randomly constrained learning.
"""
__docformat__ = 'numpy'
from time import perf_counter
from typing import NamedTuple, Sequence, Callable
from inspect import currentframe

from numpy import array as nparr, ndarray, format_float_scientific,\
                  format_float_positional
from numpy.typing import DTypeLike

from jax import grad, jit
from jax.numpy import log, argsort, cumsum, flip, argmin, argmax
from jax.lax import scan
from jax.random import split, key, permutation

from optax import adam, apply_updates
from rs import Resampler

from rcl.util import tag_logger


_default_tab_len = 12

def _lj(s: object,tab_len:int|None=_default_tab_len)-> str:
    """Left justified formatter"""
    t = str(s)
    if tab_len is None:
        return t
    if len(t) < tab_len - 1: # pad if there is space to do so
        t=' '+t
    return t.ljust(tab_len)

def _f_sc(x : DTypeLike, precision: int = 2,
          lj: int|None = _default_tab_len)-> str:
    """Scientific notation number formatting"""
    if precision:
        opp = 1+precision
        if abs(log(abs(x))/log(10))>opp:
            s=format_float_scientific(x, precision=precision)
        else:
            s=format_float_positional(x, precision=opp, unique=False,
                                      fractional=False, trim='0')
        return _lj(s, tab_len = lj)
    else:
        return _lj(x, tab_len = lj)

def _shuffle_and_batch(k, x, y, bs):
    """
    Shuffle and batch a dataset.

    Parameters
    ----------
    k : prngkey
        Jax key representing random state
    x : ndarray
        Dataset features
    y : ndarray
        Dataset targets
    bs : int
        Batch size

    Returns
    ----------
    (x_b,y_b) : tuple[ndarray,ndarray]
        Shuffled and batched dataset
    """
    ly = len(y)
    shuff = permutation(k, ly)
    x, y = x[shuff], y[shuff]
    n_batches = ly // bs
    last = n_batches * bs
    return x[0:last].reshape(n_batches, bs, - 1),\
           y[0:last].reshape(n_batches, bs)


_shuffle_and_batch = jit(_shuffle_and_batch, static_argnames='bs')


def _format_tuple(t : Sequence[object])-> str:
    """Format a sequence concisely"""
    try:
        return '(' + (','.join((str(e) for e in t))) + ')'
    except Exception as e:
        return '(!Unable to format argument as tuple: '+str(e)+'!)'


def _format_trunc_list(strs: Sequence[str], trunc: int|None, nl: str = '\n',
                       c: str=',')-> str:
    """ Format and truncate a list of strings """
    if trunc and len(strs)>trunc:
        strs=strs[:(trunc + 1) // 2] + ['...'] + strs[-trunc // 2:]
    return nl + c.join(strs) + nl


def _format_typed_list(arrays: Sequence[Sequence[object]], labels, trunc,
                       lambdas=None, nl=False)-> str:
    nl = '\n' if nl else ''
    """ Format a list according to specified types """
    if lambdas is None:
        lambdas = [str for _ in arrays]
    elif not isinstance(lambdas, Sequence):
        lambdas = [lambdas for _ in arrays]
    entries = [_format_tuple([lam(x) for lam, x in zip(lambdas, ll)])
               for ll in zip(*arrays)]
    return nl + _format_tuple(labels) + ':' +\
           _format_trunc_list(entries, trunc, nl = nl)

def _err_nature(errs):
    return '\ntype(errs):'+str(type(errs))+' elements:'+_format_tuple(errs)

class Errors:
    """
    Base class for errors.

    For binary classification an implementation needs to represent the two
    possible error types, generally this could get much more complicated.
    """
    e: ndarray
    """Errors in a raw numpy array"""

    def __init__(self, errs: [object]):
        """Instantiate new instance of Errors with some type checking"""
        assert len(errs) == len(self.error_types),\
               'Error arrays not in correspondence with error types!'+\
               _err_nature(errs)
        if all(isinstance(e,float|int) for e in errs):
            errs=nparr(errs).reshape(-1,1)
        else:
            try:
                errs=nparr(errs)
            except ValueError as e:
                raise TypeError('Invalid arguments passed in instantiation'+\
                                'of '+self.__name__+' instance!') from e
        self.e = errs

    @property
    def error_types(self)->Sequence[str]:
        """Names of error types"""
        raise NotImplementedError('Need to specify error types')

    def str(self, trunc=None):
        """Format errors as a string"""
        return _format_typed_list(self.e, self.error_types, trunc)

    def __str__(self):
        return self.str(trunc=10)

    def __repr__(self):
        return '\n==' + self.__name__ + '==' + self.__str__()

    def __getitem__(self, k):
        if isinstance(k,int):
            k=slice(k,k+1)
        return type(self)(self.e[:,k])

    def __len__(self):
        return self.e.shape[1]

    def __iter__(self):
        return (self[i] for i in range(len(self)))


class BinaryErrors(Errors):
    """Binary errors"""
    error_types = ('fp', 'fn')

    @property
    def fp(self):
        """False positives"""
        return self.e[0]

    @property
    def fn(self):
        """False negatives"""
        return self.e[1]

    def cost(self,cost_rats):
        """
        Get elementwise costs.

        Parameters
        ----------
        cost_rats : float|Sequence[float]
            Cost ratios to be used to compute costs.
            If a float is passed the ratio is fixed, otherwise cost_rats
            should have the same length as the class instance

        Returns
        -------
        costs : ndarray
            A sequence of costs, one for each error.
        """
        assert isinstance(cost_rats,float) or len(cost_rats)==len(self),\
               'Method ' + str(currentframe()) + 'requires that cost_rats be'+\
               ' a float or an array with length len(self)'
        return (self.fn * cost_rats + self.fp) / (cost_rats**.5)


class ErrorFiltration(NamedTuple):
    """
    Datatype consisting of cutoffs and errors.

    These properties represent the performance of the family of classifiers
    obtained from any continuous function/trained model by varying the cutoff
    parameter above which a prediction pred=+ is made.
    """

    cutoff: ndarray[float]
    """
    The value of the threshold above which a NN inference is treated as
    predicting a positive class
    """

    e: BinaryErrors  # ndarray[ClassifierError]
    """ Error rates for NN when cutoff is fixed """

    def __getitem__(self,k):
        assert isinstance(self.e,BinaryErrors),\
               "Incorrect type for ErrorFiltration!"
        return ErrorFiltration(self.cutoff[k],self.e[k])

    def subfiltration_from_costs(self, cost_rats: Sequence[float]):
        """
        Get a subfiltration with optimally selected cutoffs for given costs.

        Parameters
        ----------
        cost_rats : Sequence[float]|None
            Cost ratios for which to select optimal cutoffs

        Returns
        -------
        ef: ErrorFiltration
            Subfiltration with optimal cutoffs for the specified costs
        """

        inds = nparr([self._get_cutoff_index(c) for c in cost_rats])
        return self[inds]


    def subfiltration_from_cutoffs(self, cutoffs: Sequence[float]):
        """
        Get a subfiltration corresponding to specified cutoffs.

        A subfiltration here just means a class instance obtained by filtering
        errors according to a selection of choices of cutoffs.

        Parameters
        ----------
        cutoffs : [float]|None
            Cutoffs to which the filtration will be "restricted"

        Returns
        -------
        ef: ErrorFiltration
            Subfiltration corresponding to the provided cutoffs
        """
        inds = nparr([argmax(self.cutoff > c) for c in cutoffs])
        return self[inds] #ErrorFiltration(self.cutoff[inds], self.e[inds])

    def _get_cutoff_index(self, cost_rat: float) -> int:
        """Find the cutoff with best expected cost for the given cost ratio"""

        return argmin(self.e.fp + cost_rat * self.e.fn)

    def costs(self, cost_rats: [float]) -> [float]:
        """
        Get the costs associated with the filtration for specified choices of
        cost_ratio
        """

        cost_rats = nparr(cost_rats)
        assert len(cost_rats) == len(self.e),\
               'Need lengths of cost_rats and the error filtration to match'+\
               'in call to '+currentframe()

        return self.e.cost(cost_rats)

    def str(self, trunc=None):
        """Represent error filtation as a string"""
        return _format_typed_list((self.cutoff, self.e),
                                  ('class cutoffs', 'error rates'), trunc)

    def __str__(self):
        return self.str(trunc=10)

    def __repr__(self):
        return '\n==ErrorFiltration==' + self.__str__()
        # return 'ErrorFiltration'+self.__str__()

    @classmethod
    def from_predictions(self, preds: ndarray[float], targs: ndarray):
        """
        Given continuous predictions and corresponding targets on a dataset, compute the error filtration

        Parameters
        ----------
        preds : ndarray
            Continuous predictions for the target value
        targs: ndarray
            Target labels

        Returns
        ----------
        errors : ErrorFiltration
            Error filtration associated with the targets and predictions
          """
        preds = preds.reshape(-1)
        targs = targs.reshape(-1)
        sort_by_preds = argsort(preds)

        targs_by_pred = targs[sort_by_preds]
        error_increment = (1 / len(targs))

        fn_rates = cumsum(targs_by_pred) * error_increment
        fp_rates = flip(cumsum(flip(~targs_by_pred))) * error_increment
        return self(cutoff = preds[sort_by_preds],\
                    e = BinaryErrors((fp_rates, fn_rates)))


class Results(NamedTuple):
    """
    Results datatype for train-val experiments.

    Given a NN used for binary classification, one has to make one of two
    possible choices based on a float-valued output.
    Typically this decision is made depending on whether the NN output is
    above some cutoff value.

    This class represents the performance of a trained NN over a range of
    costs and cutoffs.
    """

    cost_rats: ndarray[float]
    r"""
    Cost ratios used to calculate scores.

    If $\verb|fp|$ and $\verb|fn|$ are the false positive and false negative
    rates then the cost with cost ratio $\verb|cost_rat|$ is calculated
    according to
    $$
    \verb|cost|=\frac{\verb|fp|}{\sqrt{\verb|cost_rat|}}+
                \verb|fn|\sqrt{\verb|cost_rat|}.
    $$

    This measures a NN's performance for a fixed payoff between error types.
    """

    costs_train: ndarray[float]
    """Cost as calculated on training dataset"""

    costs_val: ndarray[float]
    """Cost as calculated on val dataset"""

    costs_test: ndarray[float]
    """Cost as calculated on val dataset"""

    cutoffs: ndarray[float]
    """
    Parameterises NN behaviour as a classifier.
    Positive predictions made when NN output is above this threshold.
    """

    e_train: BinaryErrors
    """
    Raw train error rates
    """

    e_val: BinaryErrors
    """
    Raw val error rates
    """

    e_test: BinaryErrors
    """
    Raw val error rates
    """

    def __iter__(self):
        return zip(self.cost_rats, self.costs_train, self.costs_val,
                   self.costs_test, self.cutoffs, self.e_train,
                   self.e_val, self.e_test)

    @classmethod
    def from_predictions(self, preds_train, targs_train, preds_val,
                         targs_val, preds_test, targs_test, cost_rats):
        """
        Convert raw inferences on train and val datasets into a series of
        scores over a range of different possible payoffs between error types.

        The training dataset inferences are used to select an optimal cutoff
        for each error type payoff specified in cost_rats.  Once cutoffs are
        fixed the corresponding costs are returned on the train and val.

        Parameters
        ----------
        preds_train : ndarray
            Continuous training predictions
        targs_train : ndarray
            Target training labels
        preds_val : ndarray
            Continuous val predictions
        targs_val : ndarray
            Target val labels
        cost_rats : ndarray
            Cost ratios against which to benchmark the predictions.

        Returns
        -------
        res : Results
            A representation of the NN's performance for the specified
            $\\verb|cost_rats|$.
        """

        f_train = ErrorFiltration.from_predictions(preds_train, targs_train)
        f_val = ErrorFiltration.from_predictions(preds_val, targs_val)
        f_test = ErrorFiltration.from_predictions(preds_test, targs_test)

        # Pick cutoffs to minimise target cost ratios on train set
        f_train = f_train.subfiltration_from_costs(cost_rats)
        # Evaluate these training-set-optimal cutoffs to val set
        f_val = f_val.subfiltration_from_cutoffs(f_train.cutoff)
        # Evaluate these training-set-optimal cutoffs to val set
        f_test = f_test.subfiltration_from_cutoffs(f_train.cutoff)
        assert len(cost_rats)==len(f_train.costs(cost_rats))==\
               len(f_val.costs(cost_rats))==len(f_train.cutoff),\
               "Lengths don't match in instantiation of "+self.__name__+'!'
        return self(cost_rats, f_train.costs(cost_rats),
                    f_val.costs(cost_rats), f_test.costs(cost_rats),
                    f_train.cutoff, f_train.e,f_val.e, f_test.e)

    def str(self, trunc=None):
        return _format_typed_list((self.cost_rats, self.costs_train,
                                   self.costs_val, self.costs_test,
                                   self.cutoffs, self.e_train,
                                   self.e_val, self.e_test),
                                 ('cost ratio', 'E(cost|train)',
                                  'E(cost|val)', 'E(cost|test)',
                                  'class cutoff', 'train errors',
                                  'val errors', 'test errors'),
                                 trunc, lambdas=(_lj, _lj, _lj, _lj,str,str))

    def __str__(self):
        return self.str(trunc=10)

    def __repr__(self):
        return '\n==Results==' + self.__str__()


class NNState(NamedTuple):
    """NN state datatype"""
    time: int
    """time - in epochs in offline case """
    optpar: object
    """Hyperparameters for stochastic gradient descent"""
    param: object
    """Model weights"""

    def ___str__(self):
        return '(time:'+str(self.time)+'optpar:'+str(self.optpar)+\
               'param:'+self.param+')'

def _tab(items: Sequence[object], tab_len:int|None, ends=False)->str:
    '''Format a table row'''
    ret = '|'.join([_lj(obj, tab_len) for obj in items])
    return ('|'+ret+'|') if ends else ret

def _sep(n: int,tab_len: int)->str:
    return _tab(['-'*tab_len]*n,tab_len = None)

class UpdateRule(NamedTuple):
    """Single update step rule datatype"""
    lr: float
    """Learning rate"""
    bs: int
    """Batch size"""
    loss_par: tuple
    """Parameter for loss function"""

    def __str__(self):
        return '(lr:' + str(self.lr) + ' bs:' + str(self.bs) + ' loss_par: ('\
               + (', '.join([_f_sc(f, precision = 5, lj = None) for\
                             f in self.loss_par]))\
               + '))'

    def __repr__(self):
        return 'UpdateRule' + self.__str__()

    @classmethod
    def header(cls, tab_len=_default_tab_len):
        """Get a header for table purposes"""
        ret = _tab(['lr','bs'], tab_len = tab_len, ends=False) + '|'
        ret += _tab(['loss_param'], tab_len = tab_len+2, ends=False)
        return ret

    @classmethod
    def sep(cls, tab_len=_default_tab_len):
        """Separator for table purposes"""
        ret = cls.header()
        return ''.join([s if s == '|' else '-' for s in ret])

    @property
    def r(self):
        """Get a row for table purposes"""
        ret = _tab([self.lr, self.bs], _default_tab_len, ends = False) + '|'
        return ret + _tab([self.loss_par], _default_tab_len + 2, ends = False)

UpdateRule._h = UpdateRule.header()
UpdateRule._s = UpdateRule.sep()


class UpdateRuleImplementation:
    """Class for implementing an update rule for fixed parametric loss"""

    def __init__(self, loss: Callable, forward: Callable, rules: dict = {},
                 log: Callable = print):
        self.loss = loss
        self.forward = forward
        self.rules = rules
        self._log = tag_logger(log,'URI')
        """
        Implement update rules.

        Parameters
        ----------
        loss: Callable
            Parametric loss function on which to perform gradient descent
        forward: Callable
            Forward pass of architecture on which to train
        rules: dict
            Dict of rules that the instance has implemented
        log: Callable
            Method for logging
        """

    def epochs(self, ur : UpdateRule, nns : NNState, x : ndarray,
               y : ndarray, n_epochs : tuple, k : key) :
        """
        Train a NN for multiple epochs.

        Parameters
        ----------
        ur : UpdateRule
            Update rule to be applied at each step
        nns : NNState
            Representation of NN - weights and optimiser state
        x : ndarray
            Training features - used to infer likely y values
        y : ndarray
            Training target labels
        n_epochs : tuple
            Sequence of integers representing epoch numbers at which
            to save the NN's state
        k : PRNGKeyArray
            Random state (only used for shuffling training data currently)

        Returns
        -------
        nn: NNState
            Trained NN
        """

        if ur not in self.rules:
            if ur.loss_par is None:
                dl = grad(lambda param, x, y:\
                          self.loss(self.forward(param, x).\
                                    reshape(-1), y,).sum())
            else:
                dl = grad(lambda param, x, y: self.loss(
                    self.forward(param, x).reshape(-1), y, ur.loss_par).sum())
            ad = adam(learning_rate=ur.lr).update

            def step(state_param, x_y):
                x, y = x_y
                s, p = state_param
                g = dl(p, x, y)
                upd, optpar = ad(g, s)
                param = apply_updates(p, upd)
                return (optpar, param), None

            def steps(nns: NNState, x_b, y_b):
                s, p = scan(step, (nns.optpar, nns.param), (x_b, y_b))[0]
                return NNState(optpar=s, param=p, time=nns.time + 1)

            self.rules[ur] = jit(steps)
        t0 = perf_counter()
        for e, r in enumerate(split(k, n_epochs), 1):
            running_msg='Running epoch', e, 'of', n_epochs, '...'
            self._log(*running_msg, end='\r')
            x_batched, y_batched = _shuffle_and_batch(r, x, y, ur.bs)
            nns = self.rules[ur](nns, x_batched, y_batched)
        t = perf_counter() - t0
        self._log(*running_msg,'... (completed the last', n_epochs,
                 'epochs in', _f_sc(t), 'seconds)    ', end='\r')
        return nns


class TrainingCheckpoint(NamedTuple):
    """Datatype to represent the way a NN was trained"""

    n_epochs: int
    """Number of epochs NN was trained for"""
    ur: UpdateRule
    """Update rule applied"""
    rs: str
    """Resampling scheme applied"""

    def __str__(self):
        return '(n_epochs:' + str(self.n_epochs) + ' ur:' + str(self.ur) +\
               ' rs:' + (str(self.rs) if self.rs else 'None') + ')'

    def __repr__(self):
        return 'TrainingCheckpoint' + self.__str__()

    @classmethod
    def header(cls, tab_len = _default_tab_len):
        """Return a header string for table generation"""
        return _lj('epochs', tab_len = tab_len) + '|' +\
                     UpdateRule.header(tab_len = tab_len) + '|' +\
                     _lj('resampler', tab_len = tab_len)


    @classmethod
    def sep(cls, tab_len = _default_tab_len):
        """Separator for table purposes"""
        h = cls.header()
        return ''.join(['|' if c == '|' else '-' for c in h])

    @property
    def r(self):
        """Return data as a row for tables"""
        ret = _lj(self.n_epochs, tab_len = _default_tab_len) + '|'
        ret += self.ur.r + '|'
        rs = 'None' if self.rs == '' else self.rs
        ret += _lj(rs,tab_len = _default_tab_len + 2)
        return ret

TrainingCheckpoint._h = TrainingCheckpoint.header()
TrainingCheckpoint._s = TrainingCheckpoint.sep()


class Outcome(NamedTuple):
    """Represent the results of applying a fixed training definition to a NN"""
    cost_trn: float
    """Calculated cost on training ds"""
    cost_val: float
    """Calculated cost on val ds"""
    cost_tst: float
    """Calculated cost on val ds"""
    fpr_train: float
    """Training false positive rate"""
    fnr_train: float
    """Training false negative rate"""
    fpr_val: float
    """Val false positive rate"""
    fnr_val: float
    """Val false negative rate"""
    fpr_test: float
    """Val false positive rate"""
    fnr_test: float
    """Val false negative rate"""
    def __lt__(self, other):
        return self.cost_trn < other.cost_trn

    def __str__(self):
        return _format_tuple([a+':'+_f_sc(b) for a,b in\
                              self._asdict().items()])

    @classmethod
    def header(cls, tab_len = _default_tab_len):
        """Return a header string for table generation"""
        return _tab(cls._fields, tab_len = tab_len)

    @classmethod
    def sep(cls, tab_len = _default_tab_len):
        """Separator for table purposes"""
        return _sep(len(cls._fields), tab_len = tab_len)

    def r(self, precision: int = 2):
        """Return data as a row for tables"""
        return _tab([_f_sc(t, precision = precision) for t in\
                     self._asdict().values()],
                    tab_len = _default_tab_len)

    def __repr__(self):
        return 'Outcome' + self.__str__()

Outcome._h = Outcome.header()
Outcome._s = Outcome.sep()


class TrainingRule(NamedTuple):
    """Defines the way a NN is trained"""
    n_epochs: tuple
    """List of epoch numbers at which to take checkpoints"""
    ur: UpdateRule
    """Update rule for each gradient descent step"""
    rs: str
    """Resampling scheme"""

    def _train(self, uri: UpdateRuleImplementation, nns: NNState,
               rs: Resampler, k: key):
        """Train a NN according to the training rule"""
        x, y = rs.get_resampled(self.rs)
        snapshots={}
        for r, current_time, next_checkpoint in\
            zip(split(k, len(self.n_epochs)),(0,) + self.n_epochs, self):
            n_epochs = next_checkpoint.n_epochs - current_time
            nns = uri.epochs(self.ur, nns, x, y, n_epochs, r)
            snapshots[next_checkpoint]=nns
        uri._log('\n...trained for',self.n_epochs[-1],'epochs in total')
        return snapshots

    def __str__(self):
        return '(n_epochs:' + _format_tuple(self.n_epochs) + ' ur:' + str(self.ur) +\
               ' rs:' + (str(self.rs) if self.rs else 'None') + ')'

    def __repr__(self):
        return 'TrainingRule' + self.__str__()

    def __iter__(self):
        return (TrainingCheckpoint(e, self.ur, self.rs) for e in self.n_epochs)
