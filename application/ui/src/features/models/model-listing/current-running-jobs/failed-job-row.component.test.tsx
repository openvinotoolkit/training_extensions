// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, screen } from '@testing-library/react';
import { getMockedQuantizeJob } from 'mocks/mock-job';
import { getMockedModel, getMockedModelArchitecture } from 'mocks/mock-model';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../../api/utils';
import { server } from '../../../../msw-node-setup';
import { MockEventSourceConstructor, resetMockEventSource } from '../../../../test-utils/mock-event-source';
import { FailedJobRow } from './failed-job-row.component';

describe('FailedJobRow', () => {
    const mockModel = getMockedModel({
        id: 'model-123',
        architecture: 'arch-123',
        name: 'My Detection Model',
    });

    const modelArchitecture = getMockedModelArchitecture({
        performanceCategory: 'Speed',
        id: mockModel.architecture,
        name: 'Custom_Object_Detection_Gen3_ATSS',
    });

    const failedJob = getMockedQuantizeJob({
        status: 'FAILED',
        error: 'Quantization failed unexpectedly',
    });

    beforeEach(() => {
        resetMockEventSource();
        server.use(
            http.get('/api/projects/{project_id}/models/{model_id}', () => {
                return HttpResponse.json(mockModel);
            })
        );
    });

    it('renders the failed badge and a dismiss button instead of the cancel button', async () => {
        render(
            <FailedJobRow
                job={failedJob}
                onDismiss={vi.fn()}
                datasetRevisions={[]}
                modelArchitectures={[modelArchitecture]}
            />
        );

        expect(await screen.findByText('Failed')).toBeVisible();
        expect(screen.queryByText(failedJob.error as string)).not.toBeInTheDocument();
        expect(screen.getByRole('button', { name: /view logs/i })).toBeVisible();
        expect(screen.getByRole('button', { name: /dismiss failed job/i })).toBeVisible();
        expect(screen.queryByRole('button', { name: /cancel job/i })).not.toBeInTheDocument();
    });

    it('does not subscribe to SSE for a failed job', async () => {
        render(<FailedJobRow job={failedJob} datasetRevisions={[]} modelArchitectures={[modelArchitecture]} />);

        expect(await screen.findByText('Failed')).toBeVisible();
        expect(MockEventSourceConstructor).not.toHaveBeenCalled();
    });

    it('calls onDismiss when the dismiss button is pressed', async () => {
        const mockDismiss = vi.fn();

        render(
            <FailedJobRow
                job={failedJob}
                onDismiss={mockDismiss}
                datasetRevisions={[]}
                modelArchitectures={[modelArchitecture]}
            />
        );

        fireEvent.click(await screen.findByRole('button', { name: /dismiss failed job/i }));

        expect(mockDismiss).toHaveBeenCalled();
    });
});
