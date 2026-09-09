// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { components } from './../src/api/openapi-spec';

type SchemaPipelineView = components['schemas']['PipelineView'];

export const getMockedPipeline = (customPipeline?: Partial<SchemaPipelineView>): SchemaPipelineView => {
    return {
        project_id: '123',
        status: 'running' as const,
        source: {
            id: 'source-id',
            name: 'source',
            source_type: 'video_file' as const,
            video_path: 'video.mp4',
            loop: false,
        },
        model: {
            id: '1',
            name: 'My amazing model',
            architecture: 'Object_Detection_TestModel',
            training_info: {
                status: 'successful' as const,
                label_schema_revision: {},
            },
            files_deleted: false,
            variants: [],
        },
        sink: {
            id: 'sink-id',
            name: 'sink',
            folder_path: 'data/sink',
            output_formats: ['image_original', 'image_with_predictions', 'predictions'] as Array<
                'image_original' | 'image_with_predictions' | 'predictions'
            >,
            rate_limit: 0.2,
            sink_type: 'folder' as const,
        },
        device: 'cpu',
        inference: {
            confidence_threshold: 0.35,
        },
        ...customPipeline,
    };
};
