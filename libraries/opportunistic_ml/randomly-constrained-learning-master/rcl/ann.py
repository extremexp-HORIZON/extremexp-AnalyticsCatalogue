from typing import Any

PR_RETS_MAX = 10

def tn(o: Any)->str:
    '''Get a truncated type name'''
    return str(type(o)).split("'")[-2].split('.')[-1]


def len_or_shape(o: Any)->None|int|tuple:
    '''Get shape, length or none if object has the appropriate properties'''
    if hasattr(o, '__len__'):
        if hasattr(o, 'shape'):
            return o.shape
    else:
        return len(o)
    return None


def get_type_inds(o: Any)->tuple[str, str, None|int|tuple[int,...]]:
    name = tn(o)
    inds = len_or_shape(o)
    if inds is not None:
        if isinstance(inds, int):
            inds_str = str(inds)
        else:
            inds_str = ', '.join(str(i) for i in inds)
        name += f'[{inds_str}]'
    return name, inds


def annotate_call(fn, msg):
    print(f'[{msg:{15}}]==================== CALL ====================')
    if hasattr(fn,'__name__'):
        print(f'[{msg:{15}}]========= with name {fn.__name__}')


def annotate_ret(ret, msg):
    print(f'[{msg:{15}}]=================== RETURN ===================')
    name, inds = get_type_inds(ret)
    ret_rep =  f'[{msg:{15}}]=======          type : {name}\n'
    if inds and isinstance(inds, int):
        ret_rep += f'[{msg:{15}}]======     with types :'
        ret_rep += '               ========\n'
        for i, r in enumerate((*ret[:PR_RETS_MAX], '...')[:inds]):
            name, _ = get_type_inds(r)
            ret_rep += f'[{msg:{15}}]========   return[{i:{2}}] : {name:}\n'
    ret_rep +=  f'[{msg:{15}}]=============================================='
    print(ret_rep)


def ann(msg:str = 'annotated')->Any:
    if not ann.mute:
        print('# New annotation:', msg)
    def after(fn):
        if ann.disable:
            return fn
        def _fn(*a, **k):
            if not ann.mute:
                annotate_call(fn, msg)
            ret = fn(*a, **k)
            if not ann.mute:
                annotate_ret(ret, msg)
            return ret
        return _fn
    return after

ann.disable = True
ann.mute = True
