// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen, within } from '@testing-library/react';
import { getMockedJob } from 'mocks/mock-job';
import { getMockedLabel } from 'mocks/mock-labels';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../../../api/utils';
import { server } from '../../../../../msw-node-setup';
import { ImportDatasetDialogStateProvider } from '../../../providers/export-import-dataset-dialog-provider.component';
import { LabelMapping } from './label-mapping.component';

const projectLabels = [
    getMockedLabel({ name: 'label-1' }),
    getMockedLabel({ name: 'label-2' }),
    getMockedLabel({ name: 'label-3' }),
];

describe('LabelMapping', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    const renderApp = (stagedDatasetId: string, datasetLabels: string[] = []) => {
        server.use(
            http.get('/api/projects/{project_id}', () => {
                return HttpResponse.json(
                    getMockedProject({
                        id: 'project-123',
                        task: {
                            labels: projectLabels,
                            task_type: 'classification',
                            exclusive_labels: false,
                        },
                    })
                );
            }),
            http.get('/api/staged_datasets/{staged_dataset_id}', () => {
                return HttpResponse.json({
                    id: stagedDatasetId,
                    format: 'unknown',
                    compressed: true,
                    ready_for_export: false,
                    ready_for_import: true,
                    size: 10,
                    metadata: {
                        num_images: 10,
                        num_frames: 8,
                        num_annotations: 10,
                        num_videos: 1,
                        num_annotated_frames: 5,
                        num_annotated_images: 9,
                        annotation_type: 'polygon',
                        labels: datasetLabels,
                    },
                });
            }),
            http.post('/api/jobs', () => {
                return HttpResponse.json(getMockedJob({ job_id: 'job-123' }), { status: 202 });
            })
        );

        render(
            <ImportDatasetDialogStateProvider>
                <LabelMapping stagedDatasetId={stagedDatasetId} />
            </ImportDatasetDialogStateProvider>
        );
    };

    it('checks include unannotated by default', async () => {
        renderApp('staged-dataset-123');

        expect(await screen.findByRole('checkbox', { name: 'include unannotated' })).toBeChecked();
    });

    it('renders mappings for dataset labels and preselects matching project labels', async () => {
        const [firstLabel, secondLabel, thirdLabel] = projectLabels;
        renderApp('staged-dataset-123', [firstLabel.name, thirdLabel.name]);

        expect(await screen.findByLabelText(`Target label for ${firstLabel.name}`)).toBeVisible();
        expect(await screen.findByLabelText(`Target label for ${thirdLabel.name}`)).toBeVisible();
        expect(screen.queryByLabelText(`Target label for ${secondLabel.name}`)).not.toBeInTheDocument();
    });

    it('shows the placeholder label for dataset labels without a matching project label', async () => {
        renderApp('staged-dataset-123', ['unknown-label']);

        const picker = await screen.findByLabelText('Target label for unknown-label');

        expect(within(picker).getByText('Select label')).toBeVisible();
    });
});
