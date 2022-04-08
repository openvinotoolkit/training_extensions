from .config import DetectionConfig
from .task import DetectionInferenceTask, DetectionTrainTask

# Load relevant extensions to registry
import mpa_tasks.extensions.datasets.mpa_det_dataset
