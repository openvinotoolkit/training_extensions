// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { usePrefetchQuery } from '@tanstack/react-query';
import { datasetViewsQueryOptions } from 'hooks/api/dataset-views.hook';
import { usePrefetchPipeline } from 'hooks/api/pipeline.hook';
import { usePrefetchModels } from 'hooks/api/use-get-models.hook';
import { usePrefetchDatasetRevisions } from 'hooks/use-get-dataset-revisions.hook';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { usePrefetchTaskModelArchitectures } from '../../hooks/api/use-get-model-architectures.hook';
import { usePrefetchTrainingDevices } from './use-get-training-devices';

export const usePrefetchTrainModelData = () => {
    const projectId = useProjectIdentifier();

    usePrefetchTaskModelArchitectures();
    usePrefetchTrainingDevices();
    usePrefetchDatasetRevisions();
    usePrefetchQuery(datasetViewsQueryOptions(projectId));
    usePrefetchModels();
    usePrefetchPipeline();
};
