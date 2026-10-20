// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, screen } from '@testing-library/react';
import { userEvent } from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { formatBytes } from '../../../shared/util';
import { StagedImportDataset } from './staged-import-dataset.component';

describe('StagedImportDataset', () => {
    const stagedDatasetId = 'test-staged-dataset-id';
    const fileName = 'dataset.zip';

    it('opens the import dialog and sets current staged id on Continue', async () => {
        const stagedDatasetSize = 1024;
        const message = 'Map labels for the uploaded dataset';
        const openDialogSpy = rs.fn();

        server.use(
            http.get('/api/staged_datasets/{staged_dataset_id}', () => {
                return HttpResponse.json({
                    id: stagedDatasetId,
                    format: 'unknown',
                    compressed: true,
                    ready_for_export: false,
                    ready_for_import: true,
                    size: stagedDatasetSize,
                });
            })
        );

        render(
            <StagedImportDataset
                message={message}
                fileName={fileName}
                stagedDatasetId={stagedDatasetId}
                primaryButtonLabel={'continue'}
                onOpen={openDialogSpy}
                deleteEntry={rs.fn()}
            />
        );

        expect(
            await screen.findByText(`Import dataset - ${fileName} - ${formatBytes(stagedDatasetSize)}`)
        ).toBeVisible();

        expect(await screen.findByText(message)).toBeVisible();

        fireEvent.click(await screen.findByRole('button', { name: /continue/i }));

        expect(openDialogSpy).toHaveBeenCalledTimes(1);
    });

    it('renders failed import state when staged dataset request returns not found', async () => {
        const apiError = 'staged dataset not found';
        const openDialogSpy = rs.fn();

        server.use(
            http.get('/api/staged_datasets/{staged_dataset_id}', () =>
                // eslint-disable-next-line @typescript-eslint/ban-ts-comment
                // @ts-expect-error
                HttpResponse.json({ detail: apiError }, { status: 404 })
            )
        );

        render(
            <StagedImportDataset
                message={'Map labels for the uploaded dataset'}
                fileName={fileName}
                stagedDatasetId={stagedDatasetId}
                primaryButtonLabel={'continue'}
                onOpen={openDialogSpy}
                deleteEntry={rs.fn()}
            />
        );

        expect(await screen.findByText('An error occurred during staged file reading')).toBeVisible();

        await userEvent.click(screen.getByText('Technical details'));

        expect(await screen.findByText(apiError)).toBeVisible();
        expect(await screen.findByRole('button', { name: /close import dataset status/i })).toBeVisible();
        expect(screen.queryByRole('button', { name: /continue/i })).not.toBeInTheDocument();
        expect(openDialogSpy).not.toHaveBeenCalled();
    });
});
