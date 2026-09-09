// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { API_BASE_URL } from '@/api';
import { i18n } from '@/i18n';
import { screen } from '@testing-library/react';
import { getMockedPipeline } from 'mocks/mock-pipeline';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { ProjectCard } from './project-card.component';
import { formatCreationDate, getProjectTypeTitle } from './util';

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
        created_at: '2026-04-17T12:58:10.502Z',
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

        expect(
            screen.getByText(new RegExp(`Created: ${formatCreationDate(mockProject.created_at)}`))
        ).toBeInTheDocument();
        expect(screen.getByText('Object detection')).toBeInTheDocument();
        expect(screen.getByText('• Labels: Cat, Dog')).toBeInTheDocument();
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

    it('displays single label correctly', async () => {
        const singleLabelProject = getMockedProject({
            task: {
                task_type: 'classification',
                exclusive_labels: true,
                labels: [{ id: 'label-1', name: 'Person', color: '#FF0000', hotkey: 'P' }],
            },
        });

        render(<ProjectCard item={singleLabelProject} projectNames={[]} />);

        expect(await screen.findByText('• Labels: Person')).toBeInTheDocument();
    });

    it('displays multiple labels separated by commas', async () => {
        const multiLabelProject = getMockedProject({
            task: {
                task_type: 'detection',
                exclusive_labels: false,
                labels: [
                    { id: 'label-1', name: 'Car', color: '#FF0000', hotkey: 'C' },
                    { id: 'label-2', name: 'Truck', color: '#00FF00', hotkey: 'T' },
                    { id: 'label-3', name: 'Bus', color: '#0000FF', hotkey: 'B' },
                ],
            },
        });

        render(<ProjectCard item={multiLabelProject} projectNames={[]} />);

        expect(await screen.findByText('• Labels: Car, Truck, Bus')).toBeInTheDocument();
    });

    it('should handle empty labels array', async () => {
        const noLabelsProject = getMockedProject({
            task: {
                task_type: 'instance_segmentation',
                exclusive_labels: false,
                labels: [],
            },
        });

        render(<ProjectCard item={noLabelsProject} projectNames={[]} />);

        expect(await screen.findByText('• Labels:')).toBeInTheDocument();
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
