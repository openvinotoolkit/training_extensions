// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { components } from '../src/api/openapi-spec';

type SchemaStagedDatasetView = components['schemas']['StagedDatasetView'];

export const getMockedStagedDataset = (overrides: Partial<SchemaStagedDatasetView> = {}): SchemaStagedDatasetView => ({
    id: 'staged-dataset-456',
    format: 'geti',
    compressed: true,
    ready_for_export: true,
    ready_for_import: false,
    size: 17702642,
    metadata: null,
    ...overrides,
});
