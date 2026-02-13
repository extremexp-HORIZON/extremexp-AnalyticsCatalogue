from __future__ import annotations
from typing import Callable, Any, Optional
from flax.nnx import Module, Sequential as _Sequential, List, Rngs
from inspect import signature


def _pf(o):
  return o.__par_facts__() if hasattr(o, '__par_facts__') else (o,)


def _sf(o):
  return o.__seq_facts__() if hasattr(o, '__seq_facts__') else (o,)


def _si(o):
  return o.__share_input__() if hasattr(o, '__share_input__') else False


def has_keyword_arg(func: Callable[..., Any], name: str) -> bool:
  """Return True if func has keyword-only arguments with the given name."""
  return any(param.name == name and param.kind in\
             (param.KEYWORD_ONLY, param.POSITIONAL_OR_KEYWORD) for\
             param in signature(func).parameters.values())


class Composable(Module):
  '''
  Modules that we can compose.
  '''


  def __par_facts__(self):
    return (self,)


  def __seq_facts__(self):
    return (self,)


  def __share_input__(self):
    return False


  def __rshift__(self, other)->Sequential:
    return Sequential(*_sf(self), *_sf(other))


  def __lshift__(self, other)->Sequential:
    return Sequential(*_sf(other), *_sf(self))


  def __or__(self, other)->Parallel:
    return Parallel(*_pf(self), *_pf(other))


  def __and__(self, other)->Parallel:
    return Parallel(*_pf(self), *_pf(other), share_input = True)


class Fun(Composable):
  def __init__(self, f:Callable):
    self.call = f

  def __call__(self, *a, **k):
    return self.call(*a, **k)


class Sequential(_Sequential, Composable):
  def __seq_facts__(self):
    return self.layers


class Parallel(Composable):
  """A Module that applies a sequence of callables in parallel.

  This class provides a way to store and manipulate a sequence of callables
  (e.g. layers, activation functions) and apply them in parallel.

  """

  def __init__(self, *fns: Callable[..., Any],
               share_input: bool = False):
    """
    Args:
      *fns: A sequence of callables to apply.
    """
    self.factors = List(fns)
    self.si = share_input

  def __share_input__(self):
    return self.si

  @property
  def n_factors(self):
    return len(self.factors)

  def __call__(self, *args, rngs: Optional[Rngs] = None,
               **kwargs) -> tuple[Any, ...]:
    n_factors = self.n_factors
    if not n_factors:
      return ()
    elif n_factors == 1:
      return self.factors[0](*args, **kwargs)

    rngs_d = {} if rngs is None else {'rngs':rngs}

    if _si(self):
      ret = tuple([self._call_factor(f, args, kwargs, rngs_d) for\
                   f in self.factors])

    else:
      n_args = len(args)
      if  n_args!=n_factors:
        if n_args!=1:
          if n_factors != 1:
            raise ValueError(f'Wrong number of inputs {n_args} passed to ' +\
                             f'parallel product with {n_factors} factors.')
        else:
          passed_pos = args[0]
          if not isinstance(passed_pos, tuple) or len(passed_pos) != n_factors:
            raise ValueError('Wrong single argument type ' +\
                             f'{type(passed_pos)} passed to parallel ' +\
                             f'product with {n_factors}>1 factors.')
      else:
        passed_pos = args

      ret = tuple([self._call_factor(f, arg, kwargs, rngs_d) for arg, f in\
                   zip(passed_pos, self.factors)])

    return ret

  def __par_facts__(self):
    return self.layers


  @staticmethod
  def _call_factor(f, args, kwargs, rngs_d)->Any:
    if not callable(f):
      raise TypeError(f'Parallel.factor is not callable: {f}')
    rngs_f = rngs_d if has_keyword_arg(f, 'rngs') else {}
    if isinstance(args, dict):
      ret = f(**args, **kwargs, **rngs_f)
    elif isinstance(args, tuple):
      ret = f(*args, **kwargs, **rngs_f)
    else:
      ret = f(args, **kwargs, **rngs_f)
    return ret

