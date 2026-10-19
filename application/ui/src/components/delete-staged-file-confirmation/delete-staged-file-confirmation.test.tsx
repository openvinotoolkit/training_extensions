// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, screen, waitFor } from '@testing-library/react';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../api/utils';
import { server } from '../../msw-node-setup';
import { DeleteStagedFileConfirmation } from './delete-staged-file-confirmation.component';

describe('DeleteStagedFileConfirmation', () => {
    const stagedDatasetId = 'test-staged-dataset-id';

    beforeEach(() => {
        rs.clearAllMocks();
    });

    it('calls delete API and removes staged dataset from localStorage on success', async () => {
        const deleteSpy = rs.fn();
        const mockedDeleteImportEntry = rs.fn();

        server.use(
            http.delete('/api/staged_datasets/{staged_dataset_id}', () => {
                deleteSpy();
                return HttpResponse.json(null, { status: 204 });
            })
        );

        render(
            <DeleteStagedFileConfirmation stagedDatasetId={stagedDatasetId} deleteEntry={mockedDeleteImportEntry} />
        );

        fireEvent.click(screen.getByRole('button', { name: /delete import dataset status/i }));
        fireEvent.click(await screen.findByRole('button', { name: /^delete$/i }));

        await waitFor(() => {
            expect(deleteSpy).toHaveBeenCalled();
            expect(mockedDeleteImportEntry).toHaveBeenCalled();
        });
    });

    it('does not remove staged dataset from localStorage when delete API fails', async () => {
        const deleteSpy = rs.fn();
        const mockedDeleteImportEntry = rs.fn();

        server.use(
            http.delete('/api/staged_datasets/{staged_dataset_id}', () => {
                deleteSpy();
                // eslint-disable-next-line @typescript-eslint/ban-ts-comment
                // @ts-expect-error
                return HttpResponse.json({ detail: 'Something went wrong' }, { status: 500 });
            })
        );

        render(
            <DeleteStagedFileConfirmation stagedDatasetId={stagedDatasetId} deleteEntry={mockedDeleteImportEntry} />
        );

        fireEvent.click(screen.getByRole('button', { name: /delete import dataset status/i }));
        fireEvent.click(await screen.findByRole('button', { name: /^delete$/i }));

        await waitFor(() => {
            expect(deleteSpy).toHaveBeenCalled();
        });

        expect(mockedDeleteImportEntry).not.toHaveBeenCalled();
    });

    it('removes staged dataset from localStorage when delete API indicates staged file is invalid', async () => {
        const deleteSpy = rs.fn();
        const mockedDeleteImportEntry = rs.fn();

        server.use(
            http.delete('/api/staged_datasets/{staged_dataset_id}', () => {
                deleteSpy();
                // eslint-disable-next-line @typescript-eslint/ban-ts-comment
                // @ts-expect-error
                return HttpResponse.json({ detail: 'Staged dataset not found' }, { status: 404 });
            })
        );

        render(
            <DeleteStagedFileConfirmation stagedDatasetId={stagedDatasetId} deleteEntry={mockedDeleteImportEntry} />
        );

        fireEvent.click(screen.getByRole('button', { name: /delete import dataset status/i }));
        fireEvent.click(await screen.findByRole('button', { name: /^delete$/i }));

        await waitFor(() => {
            expect(deleteSpy).toHaveBeenCalled();
            expect(mockedDeleteImportEntry).toHaveBeenCalled();
        });
    });
});
