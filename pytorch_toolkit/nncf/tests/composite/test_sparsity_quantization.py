from nncf.dynamic_graph import patch_torch_operators
from nncf.algo_selector import create_compression_algorithm
from nncf.composite_compression import CompositeCompressionAlgorithm
from nncf.utils import get_all_modules_by_type
from nncf.sparsity.rb.layers import RBSparsifyingWeight
from nncf.quantization.layers import SymmetricQuantizer
from nncf.operations import UpdateWeight, UpdateInputs
from tests.test_helpers import BasicConvTestModel
from nncf.config import Config
from nncf.dynamic_graph import reset_context

patch_torch_operators()


def get_basic_sparsity_plus_quantization_config(input_sample_size=(1, 1, 4, 4)):
    config = Config()
    config.update({
        "input_info":
            {
                "sample_size": input_sample_size,
            },
        "compression": [
            {
                "algorithm": "rb_sparsity",
            },
            {
                "algorithm": "quantization",
            }
        ]
    })
    return config


def test_can_quantize_inputs_for_sparsity_plus_quantization():
    reset_context('orig')
    reset_context('quantized_graphs')
    reset_context('test')
    model = BasicConvTestModel()
    config = get_basic_sparsity_plus_quantization_config()
    compression_algo = create_compression_algorithm(model, config)
    assert isinstance(compression_algo, CompositeCompressionAlgorithm)
    sparse_quantized_model = compression_algo.model

    sparse_quantized_model_conv = get_all_modules_by_type(sparse_quantized_model, 'NNCFConv2d')

    nncf_module = next(iter(sparse_quantized_model_conv.values()))
    assert len(nncf_module.pre_ops) == 3  # 1x weight sparsifier + 1x weight quantizer + 1x input quantizer
    assert isinstance(nncf_module.pre_ops['0'], UpdateWeight)
    assert isinstance(nncf_module.pre_ops['0'].op, RBSparsifyingWeight)

    assert isinstance(nncf_module.pre_ops['1'], UpdateWeight)
    assert isinstance(nncf_module.pre_ops['1'].op, SymmetricQuantizer)

    assert isinstance(nncf_module.pre_ops['2'], UpdateInputs)
    assert isinstance(nncf_module.pre_ops['2'].op, SymmetricQuantizer)
