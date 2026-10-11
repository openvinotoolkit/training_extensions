// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { API_BASE_URL } from '@/api';
import { i18n } from '@/i18n';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { getMockedPipeline } from 'mocks/mock-pipeline';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';
import { localISOString } from 'test-utils/local-iso-string';
import { render } from 'test-utils/render';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { ProjectCard } from './project-card.component';
import { getProjectTypeTitle } from './util';

describe('ProjectCard', () => {
    const mockProject = getMockedProject({
        id: 'test-project-id',
        name: 'Test Project',
        task: {
            task_type: 'detection',
            exclusive_labels: false,
            labels: [
                { id: 'label-1', name: 'Cat', color: '#FF0000' },
                { id: 'label-2', name: 'Dog', color: '#00FF00' },
            ],
        },
        created_at: localISOString(2026, 4, 17, 12, 58),
    });

    beforeEach(() => {
        server.use(
            http.get('/api/projects/{project_id}/pipeline', () => {
                return HttpResponse.json(getMockedPipeline({ status: 'idle' }));
            })
        );
    });

    it('renders all elements correctly', async () => {
        render(<ProjectCard item={mockProject} projectNames={[]} />);

        expect(await screen.findByRole('heading', { name: 'Test Project' })).toBeInTheDocument();

        const thumbnail = await screen.findByRole('img', { name: 'Test Project' });
        expect(thumbnail).toBeInTheDocument();
        expect(thumbnail).toHaveAttribute('alt', 'Test Project');
        expect(thumbnail).toHaveAttribute('src', `${API_BASE_URL}/api/projects/test-project-id/thumbnail`);

        expect(screen.getByText(new RegExp('Created: April 17, 2026 at 12:58 PM'))).toBeInTheDocument();
        expect(screen.getByText('Object detection')).toBeInTheDocument();
        expect(screen.getByRole('button', { name: /view project labels/i })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: /open project options/i })).toBeInTheDocument();
    });

    it('shows active tag when pipeline is running', async () => {
        render(<ProjectCard item={{ ...mockProject, active_pipeline: true }} projectNames={[]} />);

        expect(await screen.findByText('Active')).toBeInTheDocument();
    });

    it('does not show active tag when pipeline is idle', async () => {
        render(<ProjectCard item={mockProject} projectNames={[]} />);

        expect(screen.queryByText('Active')).not.toBeInTheDocument();
    });

    it('renders as a link to project dataset page', async () => {
        render(<ProjectCard item={mockProject} projectNames={[]} />);

        const cardLink = await screen.findByRole('link');
        expect(cardLink).toHaveAttribute('href', '/projects/test-project-id/dataset');
    });

    it('shows all labels in a dialog when the labels button is clicked', async () => {
        const user = userEvent.setup();

        render(<ProjectCard item={mockProject} projectNames={[]} />);

        await user.click(await screen.findByRole('button', { name: /view project labels/i }));

        expect(await screen.findByRole('heading', { name: 'Labels' })).toBeInTheDocument();
        expect(screen.getByText('Cat')).toBeInTheDocument();
        expect(screen.getByText('Dog')).toBeInTheDocument();
    });

    it('should not render the labels button when there are no labels', async () => {
        const noLabelsProject = getMockedProject({
            task: {
                task_type: 'instance_segmentation',
                exclusive_labels: false,
                labels: [],
            },
        });

        render(<ProjectCard item={noLabelsProject} projectNames={[]} />);

        expect(await screen.findByRole('heading', { name: noLabelsProject.name })).toBeInTheDocument();
        expect(screen.queryByRole('button', { name: /view project labels/i })).not.toBeInTheDocument();
    });

    it('displays classification task type', async () => {
        const classificationProject = getMockedProject({
            task: { ...mockProject.task, task_type: 'classification', exclusive_labels: true },
        });

        render(<ProjectCard item={classificationProject} projectNames={[]} />);

        expect(await screen.findByText('Classification')).toBeInTheDocument();
    });

    it('displays multi-label classification task type', async () => {
        const classificationProject = getMockedProject({
            task: { ...mockProject.task, task_type: 'classification', exclusive_labels: false },
        });

        render(<ProjectCard item={classificationProject} projectNames={[]} />);

        expect(await screen.findByText('Multi-label classification')).toBeInTheDocument();
    });
});

describe('getProjectTypeTitle', () => {
    it('returns undefined when task is missing', () => {
        expect(getProjectTypeTitle(undefined, i18n.t)).toBeUndefined();
    });

    it('returns multi-label classification for non-exclusive classification tasks', () => {
        expect(
            getProjectTypeTitle(
                {
                    task_type: 'classification',
                    exclusive_labels: false,
                    labels: [],
                },
                i18n.t
            )
        ).toBe('Multi-label classification');
    });
});
