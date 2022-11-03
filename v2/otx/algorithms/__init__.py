# from .anomaly.classification.task import AnomalyClsTask
# from .anomaly.detection.task import AnomalyDetTask
# from .anomaly.segmentation.task import AnomalySegTask


# from .classification.task import ClsTask
# from .segmentation.task import SegTask

# def find_task_impl(task_type, train_type = None):
#     if task_type == "anomaly-classification":
#         return AnomalyClsTask
#     elif task_type == "anomaly-detection":
#         return AnomalyDetTask
#     elif task_type == "anomaly-segmentation":
#         return AnomalySegTask
#     elif task_type == "classification":
#         if train_type == "class-incremental":
#             return ClassIncrClsTask
#         elif train_type == "semi-sl":
#             return SemiSLTask
#         elif train_type is None:
#             return ClsTask
#         else:
#             return None
#     elif task_type == "segmantation":
#         return SegTask
#     else:
#         return None


# __all__ = [ 
#     "AnomalyClsTask",
#     "AnomalyDetTask",
#     "AnomalySegTask",
#     "find_task_impl"
# ]
