from otx.v2.api.core import AutoRunner
from tests.v2.integration.test_helper import TASK_CONFIGURATION

output_dir = "./otx-workspace/OTX-API-test"
task = "segmentation"

task_configuration = TASK_CONFIGURATION[task]
auto_runner = AutoRunner(
    work_dir=output_dir,
    task=task,
    train_data_roots=task_configuration["train_data_roots"],
    val_data_roots=task_configuration["val_data_roots"],
    test_data_roots=task_configuration["test_data_roots"],
)

results = auto_runner.train(
    max_epochs=1,
)
assert "model" in results
assert "checkpoint" in results
assert isinstance(results["checkpoint"], str)

# Validation
auto_runner.validate()

# Test
auto_runner.test()

# Prediction with single image
auto_runner.predict(
    model=results["model"],
    checkpoint=results["checkpoint"],
    img=task_configuration["sample"],
)

# Export Openvino IR Model
export_output = auto_runner.export(
    checkpoint=results["checkpoint"],
)
