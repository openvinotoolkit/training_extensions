#!/bin/bash
# otx optimize --work_dir otx-workspace/maskrcnn_swint/vitens_coliform/1 --checkpoint otx-workspace/maskrcnn_swint/vitens_coliform/1/.latest/export/exported_model.xml
# otx optimize --work_dir otx-workspace/maskrcnn_swint/vitens_coliform/2 --checkpoint otx-workspace/maskrcnn_swint/vitens_coliform/2/.latest/export/exported_model.xml
# otx optimize --work_dir otx-workspace/maskrcnn_swint/vitens_coliform/3 --checkpoint otx-workspace/maskrcnn_swint/vitens_coliform/3/.latest/export/exported_model.xml
# otx optimize --work_dir otx-workspace/maskrcnn_swint/vitens_coliform/4 --checkpoint otx-workspace/maskrcnn_swint/vitens_coliform/4/.latest/export/exported_model.xml

otx test --work_dir otx-workspace/maskrcnn_swint/vitens_coliform/1 --checkpoint otx-workspace/maskrcnn_swint/vitens_coliform/1/.latest/optimize/optimized_model.xml
otx test --work_dir otx-workspace/maskrcnn_swint/vitens_coliform/2 --checkpoint otx-workspace/maskrcnn_swint/vitens_coliform/2/.latest/optimize/optimized_model.xml
otx test --work_dir otx-workspace/maskrcnn_swint/vitens_coliform/3 --checkpoint otx-workspace/maskrcnn_swint/vitens_coliform/3/.latest/optimize/optimized_model.xml
otx test --work_dir otx-workspace/maskrcnn_swint/vitens_coliform/4 --checkpoint otx-workspace/maskrcnn_swint/vitens_coliform/4/.latest/optimize/optimized_model.xml