import itertools
from collections import OrderedDict
from typing import List, Dict, Union

import os
import shutil
import torch
from torch import Tensor, nn
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader

from nncf.config import Config
from nncf.debug import is_debug
from nncf.dynamic_graph.context import no_nncf_trace
from nncf.utils import is_main_process, in_scope_list, get_all_modules_by_type
from nncf.nncf_logger import logger as nncf_logger
from nncf.nncf_network import CompressionModuleType, NNCFNetwork
from nncf.quantization.layers import QUANTIZATION_MODULES
from nncf.quantization.hessian_trace import HessianTraceEstimator


class ManualPrecisionInitializer:
    def __init__(self, algo: 'QuantizationController', config: Config,
                 default_activation_bitwidth: int, default_weight_bitwidth: int,
                 criterion: _Loss, data_loader: DataLoader, is_distributed: bool = False):
        self._algo = algo
        self._model = self._algo._model  # type: NNCFNetwork
        self._bitwidth_per_scope = config.get('bitwidth_per_scope', {})  # type: List[List]
        self._default_activation_bitwidth = default_activation_bitwidth
        self._default_weight_bitwidth = default_weight_bitwidth
        self._criterion = criterion
        self._data_loader = data_loader
        self._is_distributed = is_distributed

        self._all_quantizations = {}
        self._ordered_weight_quantizations = []
        for class_type in QUANTIZATION_MODULES.registry_dict.values():
            quantization_type = class_type.__name__
            act_module_dict = self._model.get_compression_modules_by_type(
                CompressionModuleType.ACTIVATION_QUANTIZER)
            func_module_dict = self._model.get_compression_modules_by_type(CompressionModuleType.FUNCTION_QUANTIZER)
            weight_module_dict = self._model.get_nncf_wrapped_model()
            self._all_quantizations.update(get_all_modules_by_type(act_module_dict, quantization_type))
            self._all_quantizations.update(get_all_modules_by_type(func_module_dict, quantization_type))
            ops_quantizations = get_all_modules_by_type(weight_module_dict, quantization_type)
            self._ordered_weight_quantizations.extend([q for q in ops_quantizations.values() if q.is_weights])
            self._all_quantizations.update(ops_quantizations)

    def apply_init(self):
        for pair in self._bitwidth_per_scope:
            if len(pair) != 2:
                raise ValueError('Invalid format of bitwidth per scope: [int, str] is expected')
            bitwidth = pair[0]
            scope_name = pair[1]
            is_matched = False
            for scope, quantizer in self._all_quantizations.items():
                if in_scope_list(str(scope), scope_name):
                    quantizer.num_bits = bitwidth
                    is_matched = True
            if not is_matched:
                raise ValueError(
                    'Invalid scope name `{}`, failed to assign bitwidth {} to it'.format(scope_name, bitwidth))


class HAWQPrecisionInitializer(ManualPrecisionInitializer):
    def __init__(self, algo: 'QuantizationController', config: Config,
                 default_activation_bitwidth: int, default_weight_bitwidth: int, criterion: _Loss,
                 data_loader: DataLoader, is_distributed: bool = False):
        super().__init__(algo, config, default_activation_bitwidth, default_weight_bitwidth,
                         criterion, data_loader, is_distributed)
        self._traces_per_layer_path = config.get('traces_per_layer_path', None)
        self._num_data_points = config.get('num_data_points', 200)
        self._iter_number = config.get('iter_number', 200)
        self._tolerance = config.get('tolerance', 1e-5)
        self._bits = config.get('bits', [4, 8])

    def apply_init(self):
        runner = HessianAwarePrecisionInitializeRunner(self._algo, self._model, self._data_loader,
                                                       self._num_data_points,
                                                       self._all_quantizations, self._ordered_weight_quantizations,
                                                       self._bits, self._traces_per_layer_path)
        runner.run(self._criterion, self._iter_number, self._tolerance)
        self._model.rebuild_graph()
        if self._is_distributed:
            # NOTE: Order of quantization modules must be the same on GPUs to correctly broadcast num_bits
            sorted_quantizers = OrderedDict(sorted(self._all_quantizations.items(), key=lambda x: str(x[0])))
            for quantizer in sorted_quantizers.values():  # type: BaseQuantizer
                quantizer.broadcast_num_bits()
            if is_main_process():
                str_bw = [str(element) for element in self.get_bitwidth_per_scope(sorted_quantizers)]
                nncf_logger.info('\n'.join(['\n\"bitwidth_per_scope\": [', ',\n'.join(str_bw), ']']))

    def get_bitwidth_per_scope(self, sorted_quantizers: Dict['Scope', nn.Module]) -> List[List[Union[int, str]]]:
        full_bitwidth_per_scope = []
        for scope, quantizer in sorted_quantizers.items():
            override_weight_bitwidth = quantizer.is_weights and quantizer.num_bits != self._default_weight_bitwidth
            override_act_bitwidth = not quantizer.is_weights and quantizer.num_bits != self._default_activation_bitwidth
            if override_weight_bitwidth or override_act_bitwidth:
                full_bitwidth_per_scope.append([quantizer.num_bits, str(scope)])
        return full_bitwidth_per_scope


class PrecisionInitializerFactory:
    @staticmethod
    def create(init_type: str):
        if init_type == "manual":
            return ManualPrecisionInitializer
        if init_type == "hawq":
            return HAWQPrecisionInitializer
        raise NotImplementedError


class PerturbationObserver:
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.perturbation = None
        self.numels = None

    def calc_perturbation(self, module, inputs: torch.Tensor, output: torch.Tensor):
        input_ = inputs[0] if isinstance(inputs, tuple) else inputs
        with no_nncf_trace():
            self.perturbation = torch.norm(input_ - output, p=2) ** 2
            self.numels = input_.size().numel()
            self.input_norm = torch.norm(input_, p=2) ** 2

    def reset(self):
        self.perturbation = None
        self.numels = None

    def get_observation(self):
        return self.perturbation

    def get_numels(self):
        return self.numels

    def get_input_norm(self):
        return self.input_norm


class Perturbations:
    def __init__(self):
        self._perturbations = {}  # type: Dict[int, Dict[int, Tensor]]

    def add(self, layer_id: int, bitwidth: int, perturbation: Tensor):
        if layer_id in self._perturbations:
            self._perturbations[layer_id].update({bitwidth: perturbation})
        else:
            self._perturbations[layer_id] = {bitwidth: perturbation}

    def get(self, layer_id: int, bitwidth: int) -> Tensor:
        layer_perturbations = self._perturbations[layer_id]
        return layer_perturbations[bitwidth]

    def get_all(self) -> Dict[int, Dict[int, Tensor]]:
        return self._perturbations


class TracesPerLayer:
    def __init__(self, traces_per_layer: Tensor):
        self._traces_per_layer = traces_per_layer
        self._traces_order = [i[0] for i in
                              sorted(enumerate(traces_per_layer), reverse=False, key=lambda x: x[1])]

    def get(self, index: int) -> Tensor:
        return self._traces_per_layer[index]

    def get_order_of_traces(self) -> List[int]:
        return self._traces_order

    def get_all(self) -> Tensor:
        return self._traces_per_layer


class HessianAwarePrecisionInitializeRunner:
    def __init__(self,
                 algo: 'QuantizationController',
                 model: NNCFNetwork,
                 data_loader: DataLoader,
                 num_data_points: int,
                 all_quantizations: Dict['Scope', nn.Module],
                 ordered_weight_quantizations: List[nn.Module],
                 bits: List[int],
                 traces_per_layer_path: str = ''):
        super().__init__()
        self._algo = algo
        self._model = model
        self._all_quantizations = all_quantizations
        self._weights_to_init = ordered_weight_quantizations
        self._bits = bits
        self._traces_per_layer_path = traces_per_layer_path
        self._device = next(self._model.parameters()).device
        self._data_loader = data_loader
        self._num_data_points = num_data_points

    def run(self, criterion: _Loss, iter_number=200, tolerance=1e-5):
        disabled_gradients = self.disable_quantizer_gradients(self._all_quantizations,
                                                              self._algo.quantized_weight_modules_registry, self._model)

        traces_per_layer = self._calc_traces(criterion, iter_number, tolerance)

        self.enable_quantizer_gradients(self._model, self._all_quantizations, disabled_gradients)

        num_weights = len(self._weights_to_init)
        bits_configurations = self.get_constrained_configs(self._bits, num_weights)

        perturbations, weight_observers = self.calc_quantization_noise()

        configuration_metric = self.calc_hawq_metric_per_configuration(bits_configurations, perturbations,
                                                                       traces_per_layer, self._device)

        chosen_config_per_layer = self.choose_configuration(configuration_metric, bits_configurations,
                                                            traces_per_layer.get_order_of_traces())
        self.set_chosen_config(chosen_config_per_layer)
        ordered_metric_per_layer = self.get_metric_per_layer(chosen_config_per_layer, perturbations, traces_per_layer)
        if is_debug():
            self.HAWQDump(bits_configurations, configuration_metric, perturbations,
                          weight_observers, traces_per_layer, self._bits).run()
        return ordered_metric_per_layer

    @staticmethod
    def disable_quantizer_gradients(all_quantizations: Dict['Scope', nn.Module],
                                    quantized_weight_modules_registry: Dict['Scope', torch.nn.Module],
                                    model: nn.Module) -> List[str]:
        """
        Disables gradients of all parameters, except for layers that have quantizers for weights.
        :param all_quantizations: quantizers per scope
        :param quantized_weight_modules_registry: quantizers for weights per scope
        :param model: model to access all parameters
        :return: list of names of the parameters that were originally disabled
        """
        for module in all_quantizations.values():
            module.init_stage = True
            module.disable_gradients()
        # remember gradients of quantized modules that were enabled
        gradients_to_enable = []
        for quantized_module in quantized_weight_modules_registry.values():
            for param_name, param in quantized_module.named_parameters():
                if param.requires_grad:
                    gradients_to_enable.append(param_name)
        disabled_gradients = []
        # disable all gradients, except already disabled
        for param_name, param in model.named_parameters():
            if not param.requires_grad:
                disabled_gradients.append(param_name)
            else:
                param.requires_grad = False
        # enable gradients of quantized modules that were disabled
        for quantized_module in quantized_weight_modules_registry.values():
            for param_name, param in quantized_module.named_parameters():
                if param_name in gradients_to_enable and not 'bias' in param_name:
                    param.requires_grad = True
        return disabled_gradients

    def _calc_traces(self, criterion: _Loss, iter_number: int, tolerance: float) -> TracesPerLayer:
        if self._traces_per_layer_path:
            return TracesPerLayer(torch.load(self._traces_per_layer_path))

        trace_estimator = HessianTraceEstimator(self._model, criterion, self._device, self._data_loader,
                                                self._num_data_points)
        avg_traces = trace_estimator.get_average_traces(max_iter=iter_number, tolerance=tolerance)
        return TracesPerLayer(avg_traces)

    @staticmethod
    def enable_quantizer_gradients(model: nn.Module, all_quantizations: Dict['Scope', nn.Module],
                                   disabled_gradients: List):
        """
        Enables gradients of all parameters back, except for ones that were originally disabled
        :param all_quantizations: quantizers per scope
        :param model: model to access all parameters
        :param disabled_gradients:  list of names of the parameters that were originally disabled
        """
        for param_name, param in model.named_parameters():
            if param_name not in disabled_gradients:
                param.requires_grad = True
        for module in all_quantizations.values():
            module.init_stage = False
            module.enable_gradients()

    @staticmethod
    def get_constrained_configs(bits_: List[int], num_layers: int) -> List[List[int]]:
        bits = sorted(bits_)
        m = len(bits)
        L = num_layers
        bit_configs = []
        for j in range(1, m + 1):
            for combo_bits in itertools.combinations(bits, j):
                for combo_partitions in itertools.combinations(list(range(1, L)), j - 1):
                    bit_config = []
                    prev_p = 0
                    for (p, b) in zip(combo_partitions + (L,), combo_bits):
                        bit_config += [b] * (p - prev_p)
                        prev_p = p
                    bit_configs.append(bit_config)
        return bit_configs

    def calc_quantization_noise(self) -> [Perturbations, List[PerturbationObserver]]:
        hook_handles = []
        observers = []
        for i, module in enumerate(self._weights_to_init):
            observer = PerturbationObserver(self._device)
            hook_handles.append(module.register_forward_hook(observer.calc_perturbation))
            observers.append(observer)

        perturbations = Perturbations()
        for b in self._bits:
            for wi in self._weights_to_init:
                wi.num_bits = b

            self._model.do_dummy_forward(force_eval=True)

            for i, observer in enumerate(observers):
                perturbations.add(layer_id=i, bitwidth=b, perturbation=observer.get_observation())

        for handle in hook_handles:
            handle.remove()
        return perturbations, observers

    @staticmethod
    def calc_hawq_metric_per_configuration(bits_configurations: List[List[int]], perturbations: Perturbations,
                                           traces_per_layer: TracesPerLayer, device) -> List[Tensor]:
        configuration_metric = []
        for bits_config in bits_configurations:
            hawq_metric = torch.Tensor([0]).to(device)
            for i, layer_bits in enumerate(bits_config):
                order = traces_per_layer.get_order_of_traces()[i]
                hawq_metric += traces_per_layer.get(order) * perturbations.get(layer_id=order,
                                                                               bitwidth=layer_bits)
            configuration_metric.append(hawq_metric)
        return configuration_metric

    def choose_configuration(self, configuration_metric: List[Tensor], bits_configurations: List[List[int]],
                             traces_order: List[int]) -> List[int]:
        num_weights = len(traces_order)
        ordered_config = [0] * num_weights
        median_metric = torch.Tensor(configuration_metric).to(self._device).median()
        configuration_index = configuration_metric.index(median_metric)
        bit_configuration = bits_configurations[configuration_index]
        for i, bitwidth in enumerate(bit_configuration):
            ordered_config[traces_order[i]] = bitwidth
        if is_main_process():
            nncf_logger.info('Chosen HAWQ configuration (bitwidth per weightable layer)={}'.format(ordered_config))
            nncf_logger.debug('Order of the weightable layers in the HAWQ configuration={}'.format(traces_order))
        return ordered_config

    def set_chosen_config(self, weight_bits_per_layer: List[int]):
        for wq, bits in zip(self._weights_to_init, weight_bits_per_layer):
            wq.num_bits = bits
        pairs = self._algo.get_weights_activation_quantizers_pairs()
        for pair in pairs:
            wqs, aq = pair
            aq.num_bits = max([wq.num_bits for wq in wqs])

    def get_metric_per_layer(self, chosen_config_per_layer: List[int], perturbations: Perturbations,
                             traces_per_layer: TracesPerLayer):
        metric_per_layer = []
        for i, layer_bits in enumerate(chosen_config_per_layer):
            metric_per_layer.append(traces_per_layer.get(i) * perturbations.get(i, layer_bits))
        ordered_metric_per_layer = [i[0] for i in
                                    sorted(enumerate(metric_per_layer), reverse=True, key=lambda x: x[1])]
        return ordered_metric_per_layer

    class HAWQDump:
        def __init__(self, bits_configurations: List[List[int]], configuration_metric: List[Tensor],
                     perturbations: Perturbations, weight_observers: List[PerturbationObserver],
                     traces_per_layer: TracesPerLayer, bits: List[int]):
            self._bits_configurations = bits_configurations
            self._configuration_metric = configuration_metric
            self._num_weights = len(weight_observers)
            self._perturbations = perturbations
            self._weight_observers = weight_observers

            self._dump_dir = "hawq_dumps"
            if os.path.exists(self._dump_dir):
                shutil.rmtree(self._dump_dir)
            os.makedirs(self._dump_dir, exist_ok=True)

            self._traces_order = traces_per_layer.get_order_of_traces()
            self._traces_per_layer = traces_per_layer.get_all()

            num_of_weights = []
            norm_of_weights = []
            for i in range(self._num_weights):
                order = self._traces_order[i]
                num_of_weights.append(self._weight_observers[order].get_numels())
                norm_of_weights.append(self._weight_observers[order].get_input_norm())
            self._num_weights_per_layer = torch.Tensor(num_of_weights)
            self._norm_weights_per_layer = torch.Tensor(norm_of_weights)

            bits_in_megabyte = 2 ** 23
            self._model_sizes = []
            for bits_config in self._bits_configurations:
                size = torch.sum(torch.Tensor(bits_config) * self._num_weights_per_layer).item() / bits_in_megabyte
                self._model_sizes.append(size)
            self._bits = bits

        def run(self):
            self._dump_avg_traces()
            self._dump_density_of_quantization_noise()
            self._dump_metric()
            self._dump_perturbations_ratio()

        def _dump_avg_traces(self):
            import matplotlib.pyplot as plt
            dump_file = os.path.join(self._dump_dir, 'avg_traces_per_layer')
            torch.save(self._traces_per_layer, dump_file)
            fig = plt.figure()
            fig.suptitle('Average Hessian Trace')
            ax = fig.add_subplot(2, 1, 1)
            ax.set_yscale('log')
            ax.set_xlabel('weight quantizers')
            ax.set_ylabel('average hessian trace')
            ax.plot(self._traces_per_layer.cpu().numpy())
            plt.savefig(dump_file)

        def _dump_metric(self):
            import matplotlib.pyplot as plt
            list_to_plot = [cm.item() for cm in self._configuration_metric]
            fig = plt.figure()
            fig.suptitle('Pareto Frontier')
            ax = fig.add_subplot(2, 1, 1)
            ax.set_yscale('log')
            ax.set_xlabel('Model Size (MB)')
            ax.set_ylabel('Metric value (total perturbation)')
            ax.scatter(self._model_sizes, list_to_plot, s=20, facecolors='none', edgecolors='r')
            cm = torch.Tensor(self._configuration_metric)
            cm_m = cm.median().item()
            configuration_index = self._configuration_metric.index(cm_m)
            ms_m = self._model_sizes[configuration_index]
            ax.scatter(ms_m, cm_m, s=30, facecolors='none', edgecolors='b', label='median from all metrics')
            ax.legend()
            plt.savefig(os.path.join(self._dump_dir, 'Pareto_Frontier'))
            nncf_logger.info('Distribution of HAWQ metrics: min_value={:.3f}, max_value={:.3f}, median_value={:.3f}, '
                             'median_index={}, total_number={}'.format(cm.min().item(), cm.max().item(), cm_m,
                                                                       configuration_index,
                                                                       len(self._configuration_metric)))

        def _dump_density_of_quantization_noise(self):
            noise_per_config = []  # type: List[Tensor]
            for bits_config in self._bits_configurations:
                qnoise = 0
                for i in range(self._num_weights):
                    layer_bits = bits_config[i]
                    order = self._traces_order[i]
                    qnoise += self._perturbations.get(layer_id=order, bitwidth=layer_bits)
                noise_per_config.append(qnoise)

            list_to_plot = [cm.item() for cm in noise_per_config]
            import matplotlib.pyplot as plt
            fig = plt.figure()
            fig.suptitle('Density of quantization noise')
            ax = fig.add_subplot(2, 1, 1)
            ax.set_yscale('log')
            ax.set_xlabel('Blocks')
            ax.set_ylabel('Noise value')
            ax.scatter(self._model_sizes, list_to_plot, s=20, alpha=0.3)
            ax.legend()
            plt.savefig(os.path.join(self._dump_dir, 'Density_of_quantization_noise'))

        def _dump_perturbations_ratio(self):
            import matplotlib.pyplot as plt
            fig = plt.figure()
            fig.suptitle('Quantization noise vs Average Trace')
            ax = fig.add_subplot(2, 1, 1)
            ax.set_xlabel('Blocks')
            ax.set_yscale('log')
            b = max(self._bits)
            perturb = [p[b] for p in self._perturbations.get_all().values()]
            ax.plot([p / m / n for p, m, n in zip(perturb, self._num_weights_per_layer, self._norm_weights_per_layer)],
                    label='normalized {}-bit noise'.format(b))
            ax.plot(perturb, label='{}-bit noise'.format(b))
            ax.plot(self._traces_per_layer.cpu().numpy(), label='trace')
            ax.plot([n * p for n, p in zip(self._traces_per_layer.cpu(), perturb)], label='trace * noise')
            ax.legend()
            plt.savefig(os.path.join(self._dump_dir, 'Quantization_noise_vs_Average_Trace'))
