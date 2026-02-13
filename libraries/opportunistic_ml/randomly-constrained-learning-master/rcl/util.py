"""
Utilities
"""
from typing import Any, Generator, TextIO
from pathlib import Path
from collections.abc import Iterable, Iterator, Callable, Hashable
from hashlib import sha256
from json import dumps as dumpj, loads as loadj
from inspect import signature, Parameter

from numpy import pad, nan, format_float_positional

from IPython.display import display, update_display, Markdown

from jax import Array
from jax.numpy import expand_dims, array, concat
from jax.tree import all as al, map as mp
from flax.nnx import Pytree
from orbax.checkpoint.utils import to_shape_dtype_struct

empty = Parameter.empty

def dict_to_sorted_tup(d:dict[str, Hashable])->\
tuple[tuple[str, Hashable], ...]:
    return tuple(sorted(d.items()))


def tup_to_hashstr(tup: tuple[tuple[str, Hashable], ...])->str:
    h = sha256()
    return h.hexdigest()[:10]


def to_hashstr(o: str|list|dict)->str:
    if isinstance(o, dict):
        o = tuple(sorted(o.items()))
    h = sha256()
    try:
        h.update(dumpj(o).encode())
    except TypeError:
        h.update(o)
    return h.hexdigest()[:10]


def obj_to_json(o: dict|tuple|list, to_tup = True)->str:
    if isinstance(o, dict) and to_tup:
        o = sorted(o.items())
    return dumpj(o) + '\n'


def read_obj(p: Path, h: str = None, to_dict = True)->Any:
    if h:
        p = p / f'{h}.json'
    else:
        h = p.stem
    c = p.read_text()
    assert to_hashstr(c) == h, f'Hash mismatch opening {p}.'
    o = loadj(c)
    o = lists_to_tups(o)
    if to_dict:
        o = dict(*o)
    return o


def write_obj(p: Path, o: Any, to_tup = True)->Path:
    j = obj_to_json(o, to_tup = to_tup)
    h = to_hashstr(j)
    s = p / f'{h}.json'
    s.write_text(j)
    return h


def tag_logger(lg: Callable, tag:str, logf: TextIO|None = None)->Callable:
    def lg_tagged(*a, **k):
        k['flush'] = True
        if logf and 'file' not in k:
            k['file'] = logf
        if lg_tagged.last_end: # allow printing partial lines
            ke = dict(**k)
            ke['end'] = ''
            lg(f'[{tag}]', **ke)
        lg(*a, **k)
        lg_tagged.last_end = k.pop('end', '\n')
    lg_tagged.last_end = '\n'
    return lg_tagged


def diter(keys: Iterable[str], *dicts:Iterable[dict[str, Any]])->\
Generator[tuple[str, Any, ...], None, None]:
    for k in keys:
        yield tuple((k, *(d[k] for d in dicts)))


def exp_leading(arr):
    return expand_dims(arr, 0)


def stacks_fixed(t):
    return al(mp(lambda a: ~((a-a[0]).any()), t))


def stacks_vary(t):
    return ~stacks_fixed(t)


def add_leading_axis(t):
    return mp(exp_leading, t)


def tree_trunc(a:Pytree , n: int)->bool:
    return mp(lambda t: t[:n], a)


def get_def(t: Pytree)->Pytree:
    return mp(to_shape_dtype_struct, t)


def lists_to_tups(o: object)->object|None:
    if isinstance(o, dict):
        return {k:lists_to_tups(v) for k, v in o.items()}
    elif isinstance(o, list):
        return tuple(lists_to_tups(v) for v in o)
    else:
        return o


def cp(t):
    return mp(lambda x:x, t)


def tree_conc(*arr):
    return mp(concat, arr)
def to_traced(fn: Callable)-> Callable:
    def t_fun(*a):
        a = (array(i) for i in a)
        return fn(*a)
    return t_fun


'''
class cache_returns:
    returns = {}
    def __new__(c, f:Callable):
        def f(*a, **k):
        if f in self.returns and
'''


class ensure_exists:
    created = set()
    def __new__(c, dir_fn: Callable[..., Path])->Callable[..., Path]:

        def fn(*a, **k):
            ret = dir_fn(*a, **k)
            if ret not in c.created:
                ret.mkdir(exist_ok = True, parents = True)
                c.created.add(ret)
            return ret
        return fn


def i0[T](it:Iterable[T]|Iterator[T])->T:
    return next(iter(it)) if isinstance(it, Iterable) else next(it)


def pad_and_stack(arrs: Iterable[Array], cvs: float = nan)->Array:
    d_max = max([len(arr) for arr in arrs])
    z_axes = len(i0(arrs).shape) - 1
    z_pad = tuple((0, 0) for _ in range(z_axes))
    return array([pad(arr, ((0, d_max - len(arr)),) + z_pad,
                  constant_values = cvs) for arr in arrs])


def get_args(arch):
    par = signature(arch).parameters
    nec, opt = [], []
    for p, v in par.items():
        if v.default is empty:
            nec.append(p)
        else:
            opt.append((p, v.default))
    return nec, opt


def get_typed_arch_kwargs(arch):
    par = signature(arch).parameters
    nec, opt = set(), {}
    for p, v in par.items():
        assert v.annotation is not empty, f'Missing annotation for {v}'
        if v.default is empty:
            nec.add(p)
        else:
            opt[p] = v.default
    return nec, opt


class dm:
    ids = set()
    def __new__(c, s, i):
        if i in c.ids:
            d = update_display
        else:
            c.ids.add(i)
            d = display
        d(Markdown(s), display_id = str(i))


class print_markdown:

    s = ''
    i = 0
    dis = display
    def __new__(c, *a, end = '\n', sep = ' ', flush = None):
        fmt = sep.join([str(b) for b in a])
        lines = fmt.replace('~>', '\\rightsquigarrow').split('\n')
        in_tab = False
        md = c.s
        for line in lines:
            if line and {line[0],line [-1]} == {'|'}:
                if md and not in_tab:
                    dm(md +'\n<br>\n', c.i)
                    c.i += 1
                    md = ''
                in_tab = True
            else:
                if in_tab:
                    dm(md + '\n', c.i)
                    c.i += 1
                    md = ''
                in_tab = False
            if md:
                if c.s:
                    c.s = ''
                else:
                    md += '\n' if in_tab else '\n<br>\n'
            md += line
        md += end
        dm(md, c.i)
        if end == '\n':
            c.i += 1
        elif end != '\r':
            c.s = md
            return
        c.s = ''


def md_tab(heads, rows, sort_by: str|None = None, precision:int = 4,
           include = None):
    assert rows, 'No rows!'
    if isinstance(rows[0], dict):
        rows = [[r[i] for i in heads] for r in rows]
    if include:
        heads = [h for h in heads if h in include]
    n_fields = len(heads)
    assert all(len(r) == n_fields for r in rows), 'Inconsistent row lengths!'
    if sort_by:
        rows.sort(key = lambda r: r[sort_by])
    rs = []
    for row in rows:
        r = []
        for w in row:
            r.append(format_float_positional(w, precision = precision) if\
                     isinstance(w, float) else str(w))
        rs.append(r)
    #rows = [[str(i) for i in r] for r in rows]
    rows = [heads, ['-' for _ in range(n_fields)]] + rs
    rows = [f'|{('|'.join(i for i in r))}|' for r in rows]
    return '\n'.join(rows)


def get_varying(dicts:Iterable[dict])->list:
    par_sets = {}
    for par in dicts:
        for k, v in par.items():
            if isinstance(v, dict):
                v = tuple(sorted(v.items()))
            if k in par_sets:
                par_sets[k].add(v)
            else:
                par_sets[k] = {v}
    return sorted([k for k, v in par_sets.items() if len(v)>1])
