// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, screen, waitFor } from '@testing-library/react';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../api/utils';
import { server } from '../../msw-node-setup';
import { CancelJobConfirmation } from './cancel-job-confirmation.component';

describe('CancelJobConfirmation', () => {
    const jobId = 'test-job-id';

    beforeEach(() => {
        rs.clearAllMocks();
    });

    const renderApp = (mockedOnRemove = rs.fn()) => {
        render(<CancelJobConfirmation jobId={jobId} onRemove={mockedOnRemove} />);
    };

    it('calls cancel API and removes job from localStorage on successful cancel', async () => {
        const cancelSpy = rs.fn();
        const mockedOnRemove = rs.fn();

        server.use(
            http.post('/api/jobs/{job_id}:cancel', () => {
                cancelSpy();
                return HttpResponse.json(null, { status: 204 });
            })
        );

        renderApp(mockedOnRemove);

        fireEvent.click(screen.getByRole('button', { name: /cancel job dialog/i }));
        fireEvent.click(await screen.findByRole('button', { name: /cancel job/i }));

        await waitFor(() => {
            expect(cancelSpy).toHaveBeenCalled();
            expect(mockedOnRemove).toHaveBeenCalled();
        });
    });

    it('removes job from localStorage when job not found error occurs', async () => {
        const mockedOnRemove = rs.fn();

        server.use(
            http.post('/api/jobs/{job_id}:cancel', () =>
                // eslint-disable-next-line @typescript-eslint/ban-ts-comment
                // @ts-expect-error
                HttpResponse.json({ detail: 'Job not found' }, { status: 404 })
            )
        );

        renderApp(mockedOnRemove);

        fireEvent.click(screen.getByRole('button', { name: /cancel job dialog/i }));
        fireEvent.click(await screen.findByRole('button', { name: /cancel job/i }));

        await waitFor(() => {
            expect(mockedOnRemove).toHaveBeenCalled();
        });
    });
});
