// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, screen, waitFor } from '@testing-library/react';
import { getMockedPipeline } from 'mocks/mock-pipeline';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';
import * as actualReactRouter from 'react-router' with { rstest: 'importActual' };
import { render } from 'test-utils/render';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { ProjectsListPanel } from './projects-list-panel.component';

const selectedProjectId = 'selected-project-id';
const selectedProjectName = 'Selected Project';
const otherProjectId = 'other-project-id';
const otherProjectName = 'Other Project';

const selectedProject = getMockedProject({ id: selectedProjectId, name: selectedProjectName, active_pipeline: false });
const otherProject = getMockedProject({ id: otherProjectId, name: otherProjectName, active_pipeline: false });

const mockNavigate = rs.fn();

rs.mock('react-router', () => {
    return { ...actualReactRouter, useNavigate: () => mockNavigate };
});

describe('ProjectsListPanel', () => {
    beforeEach(() => {
        mockNavigate.mockClear();
        server.use(
            http.get('/api/projects', () => HttpResponse.json([selectedProject, otherProject])),
            http.get('/api/projects/{project_id}/pipeline', () =>
                HttpResponse.json(
                    getMockedPipeline({
                        status: 'idle',
                        model: null,
                        source: null,
                    })
                )
            ),
            http.patch('/api/projects/{project_id}/pipeline', () =>
                HttpResponse.json({
                    project_id: selectedProjectId,
                    status: 'idle',
                    device: 'cpu',
                })
            )
        );
    });

    const renderPanel = (projectId: string = selectedProjectId) => {
        return render(<ProjectsListPanel />, {
            route: `/projects/${projectId}`,
        });
    };

    const openSelector = async (projectName: string = selectedProjectName) => {
        const selectorButton = await screen.findByRole('button', {
            name: new RegExp(`Selected project ${projectName}`, 'i'),
        });
        fireEvent.click(selectorButton);
        await screen.findByText('Manage projects');
    };

    it('shows an active badge on the selector button when the selected project has an active pipeline', async () => {
        server.use(
            http.get('/api/projects', () =>
                HttpResponse.json([{ ...selectedProject, active_pipeline: true }, otherProject])
            )
        );

        renderPanel();

        expect(await screen.findByText('Active')).toBeInTheDocument();
    });

    it('does not show an active badge on the selector button when the selected project has no active pipeline', async () => {
        renderPanel();

        await screen.findByRole('button', { name: new RegExp(`Selected project ${selectedProjectName}`, 'i') });
        expect(screen.queryByText('Active')).not.toBeInTheDocument();
    });

    it('rename from another project row closes selector and opens rename dialog prefilled with project name', async () => {
        renderPanel();
        await openSelector();

        const otherProjectMenuButton = screen.getByTestId(otherProjectId);
        fireEvent.click(otherProjectMenuButton);

        const renameOption = await screen.findByText('Rename');
        fireEvent.click(renameOption);

        await waitFor(() => {
            expect(screen.queryByText('Manage projects')).not.toBeInTheDocument();
        });

        const inputField = await screen.findByLabelText(/edit project name field/i);
        expect(inputField).toBeVisible();
        expect(inputField).toHaveValue(otherProjectName);
    });

    it('blocked enable pipeline from selector closes selector and shows explanation dialog', async () => {
        renderPanel();
        await openSelector();

        const otherProjectMenuButton = screen.getByTestId(otherProjectId);
        fireEvent.click(otherProjectMenuButton);

        const enableOption = await screen.findByText('Enable pipeline');
        fireEvent.click(enableOption);

        await waitFor(() => {
            expect(screen.queryByText('Manage projects')).not.toBeInTheDocument();
        });

        expect(await screen.findByText('Cannot enable pipeline')).toBeVisible();
        expect(
            await screen.findByText('Make sure you selected a model and source before enabling the pipeline.')
        ).toBeVisible();
    });

    it('delete from project row closes selector, shows confirmation, and navigates when deleting selected project', async () => {
        const deletedProjectIds: string[] = [];
        server.use(
            http.delete('/api/projects/{project_id}', ({ params }) => {
                deletedProjectIds.push(params.project_id);
                return HttpResponse.json(null, { status: 204 });
            })
        );

        renderPanel();
        await openSelector();

        const selectedProjectMenuButton = screen.getByTestId(selectedProjectId);
        fireEvent.click(selectedProjectMenuButton);

        const deleteOption = await screen.findByText('Delete');
        fireEvent.click(deleteOption);

        await waitFor(() => {
            expect(screen.queryByText('Manage projects')).not.toBeInTheDocument();
        });

        expect(
            await screen.findByText(`Are you sure you want to delete project "${selectedProjectName}"?`)
        ).toBeVisible();

        const confirmDeleteButton = await screen.findByRole('button', { name: 'Delete' });
        fireEvent.click(confirmDeleteButton);

        await waitFor(() => {
            expect(deletedProjectIds).toContain(selectedProjectId);
        });

        await waitFor(() => {
            expect(mockNavigate).toHaveBeenCalledWith('/projects', expect.anything());
        });
    });
});
