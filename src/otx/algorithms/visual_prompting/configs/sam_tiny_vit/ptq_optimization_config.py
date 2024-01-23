"""PTQ config file."""
from nncf.parameters import ModelType
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
from nncf.quantization.range_estimator import (
    AggregatorType,
    RangeEstimatorParameters,
    StatisticsCollectorParameters,
    StatisticsType,
)

advanced_parameters = AdvancedQuantizationParameters(
    activations_range_estimator_params=RangeEstimatorParameters(
        min=StatisticsCollectorParameters(
            statistics_type=StatisticsType.QUANTILE, aggregator_type=AggregatorType.MIN, quantile_outlier_prob=1e-4
        ),
        max=StatisticsCollectorParameters(
            statistics_type=StatisticsType.QUANTILE, aggregator_type=AggregatorType.MAX, quantile_outlier_prob=1e-4
        ),
    ),
)

model_type = ModelType.TRANSFORMER
