import pytest
import os
from tests import test_models
from tests.test_helpers import create_compressed_model_and_algo_for_test
from tests.test_compressed_graph import check_model_graph, QUANTIZERS, QuantizeTestCaseConfiguration

from tests.test_compressed_graph import get_basic_quantization_config


TEST_MODELS = [(("alexnet.dot", "lenet.dot"),
                (test_models.AlexNet, test_models.LeNet),
                ((1, 3, 32, 32), (1, 3, 32, 32)))]


@pytest.fixture(scope='function', params=QUANTIZERS)
def _case_config(request):
    quantization_type = request.param
    graph_dir = os.path.join('quantized', quantization_type)
    return QuantizeTestCaseConfiguration(quantization_type, graph_dir)


@pytest.mark.parametrize(
    "model_name, model_builder, input_size", TEST_MODELS
)
def test_context_independence(model_name, model_builder, input_size, _case_config):

    config = get_basic_quantization_config(_case_config.quant_type, input_sample_size=input_size[0])
    compressed_models = [create_compressed_model_and_algo_for_test(model_builder[0](), config)[0],
                         create_compressed_model_and_algo_for_test(model_builder[1](), config)[0]]

    for i, compressed_model in enumerate(compressed_models):
        check_model_graph(compressed_model, model_name[i], _case_config.graph_dir)
