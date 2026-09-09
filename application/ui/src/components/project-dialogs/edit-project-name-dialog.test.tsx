// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, screen, waitFor } from '@testing-library/react';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../api/utils';
import { server } from '../../msw-node-setup';
import { EditProjectNameDialog } from './edit-project-name-dialog.component';

describe('EditProjectNameDialog', () => {
    const projectId = 'test-project-id';
    const projectName = 'Original Project Name';

    it('disables save button when name is empty', async () => {
        render(
            <EditProjectNameDialog
                onClose={vi.fn()}
                isOpen={true}
                projectId={projectId}
                projectName={projectName}
                projectNames={[]}
            />
        );

        const input = screen.getByLabelText(/edit project name field/i);
        fireEvent.change(input, { target: { value: '' } });

        expect(screen.getByRole('button', { name: /save/i })).toBeDisabled();
        expect(screen.getByText('Project name cannot be empty')).toBeVisible();
    });

    it('disables save button when name is unchanged', () => {
        render(
            <EditProjectNameDialog
                onClose={vi.fn()}
                isOpen={true}
                projectId={projectId}
                projectName={projectName}
                projectNames={[]}
            />
        );

        expect(screen.getByRole('button', { name: /save/i })).toBeDisabled();
    });

    it('successfully updates project name and shows success toast', async () => {
        const onClose = vi.fn();
        const newName = 'Updated Project Name';

        server.use(
            http.patch('/api/projects/{project_id}', () => {
                return HttpResponse.json(getMockedProject({ id: projectId, name: newName }), { status: 200 });
            })
        );

        render(
            <EditProjectNameDialog
                onClose={onClose}
                isOpen={true}
                projectId={projectId}
                projectName={projectName}
                projectNames={[]}
            />
        );

        const input = screen.getByLabelText(/edit project name field/i);
        fireEvent.change(input, { target: { value: newName } });

        fireEvent.click(screen.getByRole('button', { name: /save/i }));

        expect(await screen.findByText('Project updated successfully')).toBeVisible();
        await waitFor(() => {
            expect(onClose).toHaveBeenCalled();
        });
    });

    it('shows error toast when updating project name fails', async () => {
        const onClose = vi.fn();
        const newName = 'Updated Project Name';
        const errorMessage = 'Failed to update project 2';

        server.use(
            http.patch('/api/projects/{project_id}', () => {
                // eslint-disable-next-line @typescript-eslint/ban-ts-comment
                // @ts-expect-error
                return HttpResponse.json({ detail: errorMessage }, { status: 500 });
            })
        );

        render(
            <EditProjectNameDialog
                onClose={onClose}
                isOpen={true}
                projectId={projectId}
                projectName={projectName}
                projectNames={[]}
            />
        );

        const input = screen.getByLabelText(/edit project name field/i);
        fireEvent.change(input, { target: { value: newName } });

        fireEvent.click(screen.getByRole('button', { name: /save/i }));

        expect(await screen.findByText(errorMessage)).toBeVisible();
        expect(onClose).not.toHaveBeenCalled();
    });

    it('disables save button and shows error when name matches an existing project name', () => {
        const existingName = 'Another Project';

        render(
            <EditProjectNameDialog
                onClose={vi.fn()}
                isOpen={true}
                projectId={projectId}
                projectName={projectName}
                projectNames={[existingName, 'Yet Another Project']}
            />
        );

        const input = screen.getByLabelText(/edit project name field/i);
        fireEvent.change(input, { target: { value: existingName } });

        expect(screen.getByRole('button', { name: /save/i })).toBeDisabled();
        expect(screen.getByText('That project name already exists')).toBeVisible();
    });

    it('duplicate name check is case-sensitive — different casing does not trigger duplicate error', () => {
        const existingName = 'Another Project';

        render(
            <EditProjectNameDialog
                onClose={vi.fn()}
                isOpen={true}
                projectId={projectId}
                projectName={projectName}
                projectNames={[existingName]}
            />
        );

        const input = screen.getByLabelText(/edit project name field/i);
        fireEvent.change(input, { target: { value: existingName.toUpperCase() } });

        expect(screen.queryByText('That project name already exists')).not.toBeInTheDocument();
    });

    it('enables save button when a valid new name is entered', () => {
        render(
            <EditProjectNameDialog
                onClose={vi.fn()}
                isOpen={true}
                projectId={projectId}
                projectName={projectName}
                projectNames={['Taken Name']}
            />
        );

        const input = screen.getByLabelText(/edit project name field/i);
        fireEvent.change(input, { target: { value: 'Brand New Name' } });

        expect(screen.getByRole('button', { name: /save/i })).not.toBeDisabled();
        expect(screen.queryByText('That project name already exists')).not.toBeInTheDocument();
    });
});
