// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { getMockedPipeline } from 'mocks/mock-pipeline';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { ImportDatasetDialogProvider } from '../providers/import-dataset-dialog-provider.component';
import { ProjectList } from './project-list.component';

const renderProjectList = () => {
    return render(
        <ImportDatasetDialogProvider>
            <ProjectList />
        </ImportDatasetDialogProvider>,
        { route: '/projects', path: '/projects' }
    );
};

const projects = [
    getMockedProject({
        id: 'project-1',
        name: 'Alpha Project',
        created_at: '2026-01-01T10:00:00Z',
        task: { exclusive_labels: true, labels: [], task_type: 'detection' },
    }),
    getMockedProject({
        id: 'project-2',
        name: 'Beta Project',
        created_at: '2026-06-01T10:00:00Z',
        task: { exclusive_labels: true, labels: [], task_type: 'classification' },
    }),
    getMockedProject({
        id: 'project-3',
        name: 'Zeta Project',
        created_at: '2026-03-01T10:00:00Z',
        task: { exclusive_labels: true, labels: [], task_type: 'instance_segmentation' },
    }),
    getMockedProject({
        id: 'project-4',
        name: 'Gamma Project',
        created_at: '2026-02-01T10:00:00Z',
        task: { exclusive_labels: true, labels: [], task_type: 'classification' },
    }),
    getMockedProject({
        id: 'project-5',
        name: 'Delta Project',
        created_at: '2026-04-01T10:00:00Z',
        task: { exclusive_labels: true, labels: [], task_type: 'detection' },
    }),
    getMockedProject({
        id: 'project-6',
        name: 'Epsilon Project',
        created_at: '2026-05-01T10:00:00Z',
        task: { exclusive_labels: true, labels: [], task_type: 'instance_segmentation' },
    }),
    getMockedProject({
        id: 'project-7',
        name: 'Theta Project',
        created_at: '2026-07-01T10:00:00Z',
        task: { exclusive_labels: true, labels: [], task_type: 'instance_segmentation' },
    }),
    getMockedProject({
        id: 'project-8',
        name: 'Iota Project',
        created_at: '2026-08-01T10:00:00Z',
        task: { exclusive_labels: true, labels: [], task_type: 'detection' },
    }),
];

describe('ProjectList', () => {
    describe('with projects', () => {
        beforeEach(() => {
            server.use(
                http.get('/api/projects', () => {
                    return HttpResponse.json(projects);
                }),
                http.get('/api/projects/{project_id}/pipeline', () => {
                    return HttpResponse.json(getMockedPipeline({ status: 'idle' }));
                })
            );
        });

        it('renders the "Projects" heading', async () => {
            renderProjectList();

            expect(await screen.findByRole('heading', { name: 'Projects' })).toBeInTheDocument();
        });

        it('renders a card for each project', async () => {
            renderProjectList();

            expect(await screen.findByRole('heading', { name: 'Alpha Project' })).toBeInTheDocument();
            expect(await screen.findByRole('heading', { name: 'Beta Project' })).toBeInTheDocument();
            expect(await screen.findByRole('heading', { name: 'Zeta Project' })).toBeInTheDocument();
        });

        it('defaults to sorting by created date (newest first)', async () => {
            renderProjectList();

            const headings = await screen.findAllByRole('heading', {
                name: /Alpha Project|Beta Project|Zeta Project/,
            });

            expect(headings[0]).toHaveTextContent('Beta Project');
            expect(headings[1]).toHaveTextContent('Zeta Project');
            expect(headings[2]).toHaveTextContent('Alpha Project');
        });

        it('sorts projects by name ascending when selected', async () => {
            const user = userEvent.setup();
            renderProjectList();

            const picker = await screen.findByRole('button', { name: /sort/i });
            await user.click(picker);

            const option = await screen.findByRole('option', { name: 'Name (A-Z)' });
            await user.click(option);

            const headings = await screen.findAllByRole('heading', {
                name: /Alpha Project|Beta Project|Zeta Project/,
            });

            expect(headings[0]).toHaveTextContent('Alpha Project');
            expect(headings[1]).toHaveTextContent('Beta Project');
            expect(headings[2]).toHaveTextContent('Zeta Project');
        });

        it('sorts projects by name descending when selected', async () => {
            const user = userEvent.setup();
            renderProjectList();

            const picker = await screen.findByRole('button', { name: /sort/i });
            await user.click(picker);

            const option = await screen.findByRole('option', { name: 'Name (Z-A)' });
            await user.click(option);

            const headings = await screen.findAllByRole('heading', {
                name: /Alpha Project|Beta Project|Zeta Project/,
            });

            expect(headings[0]).toHaveTextContent('Zeta Project');
            expect(headings[1]).toHaveTextContent('Beta Project');
            expect(headings[2]).toHaveTextContent('Alpha Project');
        });

        it('sorts projects by created date oldest first when selected', async () => {
            const user = userEvent.setup();
            renderProjectList();

            const picker = await screen.findByRole('button', { name: /sort/i });
            await user.click(picker);

            const option = await screen.findByRole('option', { name: 'Created date (oldest)' });
            await user.click(option);

            const headings = await screen.findAllByRole('heading', {
                name: /Alpha Project|Beta Project|Zeta Project/,
            });

            expect(headings[0]).toHaveTextContent('Alpha Project');
            expect(headings[1]).toHaveTextContent('Zeta Project');
            expect(headings[2]).toHaveTextContent('Beta Project');
        });

        it('each project card links to the project dataset page', async () => {
            renderProjectList();

            const alphaCard = (await screen.findByRole('heading', { name: 'Alpha Project' })).closest('a');
            expect(alphaCard).toHaveAttribute('href', '/projects/project-1/dataset');

            const betaCard = (await screen.findByRole('heading', { name: 'Beta Project' })).closest('a');
            expect(betaCard).toHaveAttribute('href', '/projects/project-2/dataset');
        });
    });

    describe('with search and task type filters', () => {
        beforeEach(() => {
            server.use(
                http.get('/api/projects', () => {
                    return HttpResponse.json(projects);
                }),
                http.get('/api/projects/{project_id}/pipeline', () => {
                    return HttpResponse.json(getMockedPipeline({ status: 'idle' }));
                })
            );
        });

        it('shows the project count label by default (not filtering)', async () => {
            renderProjectList();

            expect(await screen.findByRole('heading', { name: 'Alpha Project' })).toBeInTheDocument();

            expect(screen.getByText('8 projects')).toBeInTheDocument();
            expect(screen.queryByText(/of 8 projects/i)).not.toBeInTheDocument();
        });

        it('filters projects by name search', async () => {
            const user = userEvent.setup();
            renderProjectList();

            expect(await screen.findByRole('heading', { name: 'Alpha Project' })).toBeInTheDocument();

            await user.type(screen.getByRole('searchbox', { name: /search projects by name/i }), 'beta');

            expect(await screen.findByText('1 of 8 projects')).toBeInTheDocument();
            expect(screen.getByRole('heading', { name: 'Beta Project' })).toBeInTheDocument();
            expect(screen.queryByRole('heading', { name: 'Alpha Project' })).not.toBeInTheDocument();
            expect(screen.queryByRole('heading', { name: 'Zeta Project' })).not.toBeInTheDocument();
        });

        it('filters projects by task type', async () => {
            const user = userEvent.setup();
            renderProjectList();

            expect(await screen.findByRole('heading', { name: 'Alpha Project' })).toBeInTheDocument();

            await user.click(screen.getByRole('button', { name: 'Filter by task type' }));

            const classificationCheckbox = await screen.findByRole('checkbox', { name: 'Classification' });
            await user.click(classificationCheckbox);

            expect(classificationCheckbox).toBeChecked();

            await user.keyboard('{Escape}');

            expect(await screen.findByText('1 type selected')).toBeInTheDocument();
            expect(await screen.findByText('2 of 8 projects')).toBeInTheDocument();
            expect(screen.getByRole('heading', { name: 'Beta Project' })).toBeInTheDocument();
            expect(screen.getByRole('heading', { name: 'Gamma Project' })).toBeInTheDocument();
            expect(screen.queryByRole('heading', { name: 'Alpha Project' })).not.toBeInTheDocument();
            expect(screen.queryByRole('heading', { name: 'Zeta Project' })).not.toBeInTheDocument();
        });

        it('filters projects by both name search and task type together', async () => {
            const user = userEvent.setup();
            renderProjectList();

            expect(await screen.findByRole('heading', { name: 'Alpha Project' })).toBeInTheDocument();

            await user.click(screen.getByRole('button', { name: 'Filter by task type' }));
            await user.click(await screen.findByRole('checkbox', { name: 'Object detection' }));
            await user.keyboard('{Escape}');

            await user.type(screen.getByRole('searchbox', { name: /search projects by name/i }), 'alpha');

            expect(await screen.findByText('1 of 8 projects')).toBeInTheDocument();
            expect(screen.getByRole('heading', { name: 'Alpha Project' })).toBeInTheDocument();
            expect(screen.queryByRole('heading', { name: 'Beta Project' })).not.toBeInTheDocument();
            expect(screen.queryByRole('heading', { name: 'Zeta Project' })).not.toBeInTheDocument();
        });

        it('shows no matching projects when the name search and task type filters do not overlap', async () => {
            const user = userEvent.setup();
            renderProjectList();

            expect(await screen.findByRole('heading', { name: 'Alpha Project' })).toBeInTheDocument();

            await user.click(screen.getByRole('button', { name: 'Filter by task type' }));
            await user.click(await screen.findByRole('checkbox', { name: 'Object detection' }));
            await user.keyboard('{Escape}');

            await user.type(screen.getByRole('searchbox', { name: /search projects by name/i }), 'beta');

            expect(await screen.findByText('No projects match your filters')).toBeInTheDocument();
            expect(screen.getByText('0 of 8 projects')).toBeInTheDocument();
            expect(
                screen.queryByRole('heading', { name: /Alpha Project|Beta Project|Zeta Project/ })
            ).not.toBeInTheDocument();
        });

        it('shows a "no matching projects" message when filters exclude all projects', async () => {
            const user = userEvent.setup();
            renderProjectList();

            expect(await screen.findByRole('heading', { name: 'Alpha Project' })).toBeInTheDocument();

            await user.type(screen.getByRole('searchbox', { name: /search projects by name/i }), 'does-not-exist');

            expect(await screen.findByText('No projects match your filters')).toBeInTheDocument();
        });
    });

    describe('with exactly one non-active project (boundary)', () => {
        beforeEach(() => {
            server.use(
                http.get('/api/projects', () => {
                    return HttpResponse.json(projects.slice(0, 1));
                }),
                http.get('/api/projects/{project_id}/pipeline', () => {
                    return HttpResponse.json(getMockedPipeline({ status: 'idle' }));
                })
            );
        });

        it('shows the filters and a singular project count label', async () => {
            renderProjectList();

            expect(await screen.findByRole('heading', { name: 'Alpha Project' })).toBeInTheDocument();

            expect(screen.getByRole('button', { name: /sort/i })).toBeInTheDocument();
            expect(screen.getByRole('button', { name: 'Filter by task type' })).toBeInTheDocument();
            expect(screen.getByRole('searchbox', { name: /search projects by name/i })).toBeInTheDocument();
            expect(screen.getByText('1 project')).toBeInTheDocument();
        });
    });

    describe('with only an active pipeline project (no non-active projects)', () => {
        const onlyActiveProject = getMockedProject({
            id: 'project-active-only',
            name: 'Running Project',
            created_at: '2026-09-01T10:00:00Z',
            active_pipeline: true,
            task: { exclusive_labels: true, labels: [], task_type: 'detection' },
        });

        beforeEach(() => {
            server.use(
                http.get('/api/projects', () => {
                    return HttpResponse.json([onlyActiveProject]);
                }),
                http.get('/api/projects/{project_id}/pipeline', () => {
                    return HttpResponse.json(getMockedPipeline({ status: 'idle' }));
                })
            );
        });

        it('still pins the active project card', async () => {
            renderProjectList();

            expect(await screen.findByRole('heading', { name: 'Running Project' })).toBeInTheDocument();
        });

        it('hides the sort control', async () => {
            renderProjectList();

            expect(await screen.findByRole('heading', { name: 'Running Project' })).toBeInTheDocument();

            expect(screen.queryByRole('button', { name: /sort/i })).not.toBeInTheDocument();
        });

        it('hides the task type filter control', async () => {
            renderProjectList();

            expect(await screen.findByRole('heading', { name: 'Running Project' })).toBeInTheDocument();

            expect(screen.queryByRole('button', { name: 'Filter by task type' })).not.toBeInTheDocument();
        });

        it('hides the search box', async () => {
            renderProjectList();

            expect(await screen.findByRole('heading', { name: 'Running Project' })).toBeInTheDocument();

            expect(screen.queryByRole('searchbox', { name: /search projects by name/i })).not.toBeInTheDocument();
        });

        it('does not show a project count label', async () => {
            renderProjectList();

            expect(await screen.findByRole('heading', { name: 'Running Project' })).toBeInTheDocument();

            expect(screen.queryByText(/^\d+ projects?$/i)).not.toBeInTheDocument();
        });
    });

    describe('with an active pipeline project', () => {
        const activeProject = getMockedProject({
            id: 'project-active',
            name: 'Running Project',
            created_at: '2026-09-01T10:00:00Z',
            active_pipeline: true,
            task: { exclusive_labels: true, labels: [], task_type: 'detection' },
        });

        beforeEach(() => {
            server.use(
                http.get('/api/projects', () => {
                    return HttpResponse.json([...projects, activeProject]);
                }),
                http.get('/api/projects/{project_id}/pipeline', () => {
                    return HttpResponse.json(getMockedPipeline({ status: 'idle' }));
                })
            );
        });

        it('pins the active project card next to the "create new project" card with an "Active" badge', async () => {
            renderProjectList();

            const activeCard = await screen.findByLabelText('Project: Running Project');
            expect(activeCard).not.toBeNull();
            expect(activeCard && within(activeCard as HTMLElement).getByText('Active')).toBeInTheDocument();
        });

        it('excludes the active project from the project count', async () => {
            renderProjectList();

            expect(await screen.findByRole('heading', { name: 'Running Project' })).toBeInTheDocument();

            // 9 projects total, 1 is pinned as active, so 8 remain in the sortable/filterable list.
            expect(await screen.findByText('8 projects')).toBeInTheDocument();
        });

        it('excludes the active project from name search results', async () => {
            const user = userEvent.setup();
            renderProjectList();

            expect(await screen.findByRole('heading', { name: 'Running Project' })).toBeInTheDocument();

            await user.type(screen.getByRole('searchbox', { name: /search projects by name/i }), 'running');

            expect(await screen.findByText('No projects match your filters')).toBeInTheDocument();
        });
    });

    describe('with no projects', () => {
        beforeEach(() => {
            server.use(
                http.get('/api/projects', () => {
                    return HttpResponse.json([]);
                })
            );
        });

        it('shows empty illustration when there are no projects', async () => {
            renderProjectList();

            expect(await screen.findByLabelText('empty list')).toBeInTheDocument();
            expect(screen.getByRole('button', { name: /create new project/i })).toBeVisible();
            expect(screen.getByRole('button', { name: /create from dataset/i })).toBeVisible();

            expect(screen.queryByRole('button', { name: /sort/i })).not.toBeInTheDocument();
        });

        it('shows the Geti intro and the workflow steps', async () => {
            renderProjectList();

            expect(await screen.findByLabelText('empty list')).toBeInTheDocument();
            expect(screen.getByText(/a vision AI platform that guides you through/)).toBeInTheDocument();

            const workflow = screen.getByRole('list', { name: 'Geti workflow' });
            expect(
                within(workflow)
                    .getAllByRole('listitem')
                    .map((item) => item.textContent)
            ).toEqual(['Add data', 'Annotate', 'Train', 'Optimize', 'Run inference']);
            expect(
                screen.getByText('Monitor predictions and collect more data to iteratively fine-tune your model')
            ).toBeInTheDocument();
        });
    });

    describe('create project card', () => {
        it('renders create new project and create project from dataset buttons', async () => {
            server.use(http.get('/api/projects', () => HttpResponse.json(projects)));

            renderProjectList();

            const createButton = await screen.findByRole('button', { name: /create new project/i });
            expect(createButton).toBeVisible();

            const createFromDatasetButton = await screen.findByRole('button', { name: /Create project from dataset/i });
            expect(createFromDatasetButton).toBeVisible();
        });
    });
});
