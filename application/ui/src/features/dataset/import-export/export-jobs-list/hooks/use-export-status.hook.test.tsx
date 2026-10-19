// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { waitFor } from '@testing-library/react';
import { HttpResponse } from 'msw';
import { renderHook } from 'test-utils/render';

import { http } from '../../../../../api/utils';
import { server } from '../../../../../msw-node-setup';
import { useExportStatus } from './use-export-status.hook';

const mockRemoveLsExportId = rs.fn();

rs.mock('hooks/storage/use-export-dataset.hook', () => ({
    useExportDataset: () => ({
        removeLsExportId: mockRemoveLsExportId,
        lsExportIds: [],
        addLsExportId: rs.fn(),
    }),
}));

describe('useExportStatus', () => {
    const jobId = 'test-job-id';

    it('removes job from localStorage when job not found error occurs', async () => {
        server.use(
            http.get('/api/jobs/{job_id}', () =>
                // eslint-disable-next-line @typescript-eslint/ban-ts-comment
                // @ts-expect-error
                HttpResponse.json({ detail: 'Job not found' }, { status: 404 })
            )
        );

        renderHook(() => useExportStatus(jobId));

        await waitFor(() => {
            expect(mockRemoveLsExportId).toHaveBeenCalledWith(jobId);
        });
    });
});
