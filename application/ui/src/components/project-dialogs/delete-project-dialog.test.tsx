// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, screen, waitFor } from '@testing-library/react';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../api/utils';
import { server } from '../../msw-node-setup';
import { DeleteProjectDialog } from './delete-project-dialog.component';

describe('DeleteProjectDialog', () => {
    const projectId = 'test-project-id';
    const projectName = 'Test Project';

    it('preserves punctuation and markup-like text in the project name', () => {
        const name = 'A "quoted" project & <sample>';

        render(<DeleteProjectDialog projectId={projectId} projectName={name} isOpen onClose={vi.fn()} />);

        expect(screen.getByText(`Are you sure you want to delete project "${name}"?`)).toBeVisible();
        expect(document.querySelector('sample')).toBeNull();
    });

    it('successfully deletes project and shows success toast', async () => {
        server.use(
            http.delete('/api/projects/{project_id}', () => {
                return HttpResponse.json(null, { status: 204 });
            })
        );

        const onClose = vi.fn();
        render(<DeleteProjectDialog projectId={projectId} projectName={projectName} isOpen onClose={onClose} />);

        fireEvent.click(screen.getByRole('button', { name: 'Delete' }));

        expect(await screen.findByText('Project deleted successfully')).toBeVisible();
        await waitFor(() => {
            expect(onClose).toHaveBeenCalled();
        });
    });

    it('shows error toast when delete project fails', async () => {
        const errorMessage = 'Cannot delete project';
        server.use(
            http.delete('/api/projects/{project_id}', () => {
                // eslint-disable-next-line @typescript-eslint/ban-ts-comment
                // @ts-expect-error
                return HttpResponse.json({ detail: errorMessage }, { status: 500 });
            })
        );

        const onClose = vi.fn();
        render(<DeleteProjectDialog projectId={projectId} projectName={projectName} isOpen onClose={onClose} />);

        fireEvent.click(screen.getByRole('button', { name: 'Delete' }));

        expect(await screen.findByText(errorMessage)).toBeVisible();
        await waitFor(() => {
            expect(onClose).toHaveBeenCalled();
        });
    });
});
