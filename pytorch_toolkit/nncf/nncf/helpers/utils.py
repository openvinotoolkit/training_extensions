from torch import distributed as dist, nn

from nncf.dynamic_graph.transform_graph import replace_modules
from nncf.layers import NNCF_RNN


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def safe_thread_call(main_call_fn, after_barrier_call_fn=None):
    result = None
    if is_dist_avail_and_initialized():
        if is_main_process():
            result = main_call_fn()
        dist.barrier()
        if not is_main_process():
            result = after_barrier_call_fn() if after_barrier_call_fn else main_call_fn()
    else:
        result = main_call_fn()
    return result


def replace_lstm(model):
    def replace_fn(module_):
        if not isinstance(module_, nn.LSTM):
            return module_
        device = next(module_.parameters()).device
        custom_lstm = NNCF_RNN('LSTM', input_size=module_.input_size, hidden_size=module_.hidden_size,
                               num_layers=module_.num_layers, bidirectional=module_.bidirectional,
                               batch_first=module_.batch_first, dropout=module_.dropout,
                               bias=module_.bias)

        def get_param_names(bias):
            # type: (bool) -> List[str]
            suffixes = ['ih', 'hh']
            names = ['weight_' + suffix for suffix in suffixes]
            if bias:
                names += ['bias_' + suffix for suffix in suffixes]
            return names

        for l in range(custom_lstm.num_layers):
            for d in range(custom_lstm.num_directions):
                for name in get_param_names(custom_lstm.bias):
                    suffix = '_reverse' if d == 1 else ''
                    param_name = name + '_l{}{}'.format(l, suffix)
                    param = getattr(module_, param_name)
                    getattr(custom_lstm, param_name).data.copy_(param.data)
        custom_lstm.to(device)
        return custom_lstm

    if isinstance(model, nn.LSTM):
        return replace_fn(model)
    affected_scopes = []
    return replace_modules(model, replace_fn, affected_scopes)[0]
