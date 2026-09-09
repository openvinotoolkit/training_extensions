// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { TaskType } from '@/api/types';

import { ImportDatasetAsNewProjectState } from '../../features/dataset/import-export/import-dataset/util';
import { useDatasetImportStorage } from './use-dataset-import-storage.hook';
import { DatasetImportState } from './utils';

type DatasetImportAsNewProjectState = DatasetImportState<ImportDatasetAsNewProjectState> & {
    project?: { name: string; task_type: TaskType };
    filters?: { labels: string[]; include_unannotated: boolean };
};

export const IMPORT_DATASET_AS_NEW_PROJECT_KEY = 'import-dataset-as-new-project';

export const useImportDatasetAsNewProject = () => {
    return useDatasetImportStorage<ImportDatasetAsNewProjectState, DatasetImportAsNewProjectState>(
        IMPORT_DATASET_AS_NEW_PROJECT_KEY
    );
};
