"""Debug."""

# from tests.helpers.dataset import OTEAnomalyDatasetGenerator
from tests.helpers.train import OTEAnomalyTrainer

# dataset = OTEAnomalyDatasetGenerator(path="/home/sakcay/projects/anomalib/datasets/MVTec/bottle").generate()

trainer1 = OTEAnomalyTrainer(
    model_template_path="anomaly_classification/configs/padim/template.yaml",
    dataset_path="/home/sakcay/projects/anomalib/datasets/MVTec",
    category="bottle",
)

trainer1.train()
base_results1 = trainer1.validate(task=trainer1.base_task)

# Load the model on a new task
trainer2 = OTEAnomalyTrainer(
    model_template_path="anomaly_classification/configs/padim/template.yaml",
    dataset_path="/home/sakcay/projects/anomalib/datasets/MVTec",
    category="bottle",
)

trainer2.dataset = trainer1.dataset
trainer2.base_task.model = trainer2.base_task.load_model(trainer1.output_model)
trainer2.base_task.model = trainer2.base_task.load_model(trainer1.base_task.task_environment.model)
base_results2 = trainer2.validate(task=trainer2.base_task)

# Convert the model to OpenVINO
trainer1.export()
openvino_results = trainer1.validate(task=trainer1.openvino_task)

# Optimize the OpenVINO Model via POT
optimized_openvino_results = trainer1.validate(task=trainer1.openvino_task, optimize=True)
