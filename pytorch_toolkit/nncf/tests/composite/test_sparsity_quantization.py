from nncf.composite_compression import CompositeCompressionAlgorithmController
from nncf.utils import get_all_modules_by_type
from nncf.sparsity.rb.layers import RBSparsifyingWeight
from nncf.quantization.layers import SymmetricQuantizer
from nncf.module_operations import UpdateWeight, UpdateInputs
from tests.test_helpers import BasicConvTestModel, create_compressed_model_and_algo_for_test
from nncf.config import Config



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
                "algorithm": "quantization"
            }
        ]
    })
    return config


def test_can_quantize_inputs_for_sparsity_plus_quantization():
    model = BasicConvTestModel()
    config = get_basic_sparsity_plus_quantization_config()
    sparse_quantized_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    assert isinstance(compression_ctrl, CompositeCompressionAlgorithmController)

    sparse_quantized_model_conv = get_all_modules_by_type(sparse_quantized_model, 'NNCFConv2d')

    nncf_module = next(iter(sparse_quantized_model_conv.values()))
    assert len(nncf_module.pre_ops) == 3  # 1x weight sparsifier + 1x weight quantizer + 1x input quantizer
    assert isinstance(nncf_module.pre_ops['0'], UpdateWeight)
    assert isinstance(nncf_module.pre_ops['0'].op, RBSparsifyingWeight)

    assert isinstance(nncf_module.pre_ops['1'], UpdateWeight)
    assert isinstance(nncf_module.pre_ops['1'].op, SymmetricQuantizer)

    assert isinstance(nncf_module.pre_ops['2'], UpdateInputs)
    assert isinstance(nncf_module.pre_ops['2'].op, SymmetricQuantizer)
