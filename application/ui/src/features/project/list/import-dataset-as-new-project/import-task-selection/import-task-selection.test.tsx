// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { AnnotationType, DatasetFormat, TaskType } from '@/api/types';
import { screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../../../api/utils';
import { server } from '../../../../../msw-node-setup';
import { ImportTaskSelection } from './import-task-selection.component';

const mockedStagedDatasetId = 'staged-dataset-123';
const mockedProject = getMockedProject({ id: 'id-1', name: 'Project 1' });

const getImportEntrySpy = rs.fn();
const updateImportEntrySpy = rs.fn();
const setCurrentStepSpy = rs.fn();

rs.mock('hooks/storage/use-import-dataset-as-new-project.hook', () => ({
    useImportDatasetAsNewProject: () => ({
        getAllImportEntries: rs.fn(),
        appendImportEntry: rs.fn(),
        deleteImportEntry: rs.fn(),
        updateImportEntryStep: rs.fn(),
        updateImportEntry: updateImportEntrySpy,
        getImportEntry: getImportEntrySpy,
    }),
}));

rs.mock('../../../providers/import-dataset-dialog-provider.component', () => ({
    useImportDatasetDialog: () => ({
        setCurrentStep: setCurrentStepSpy,
    }),
}));

describe('ImportTaskSelection', () => {
    beforeEach(() => {
        rs.clearAllMocks();
    });

    const renderApp = ({
        format = 'geti',
        taskType,
        annotationType,
    }: {
        format: DatasetFormat;
        taskType?: TaskType;
        annotationType: AnnotationType;
    }) => {
        getImportEntrySpy.mockReturnValue({
            step: 'taskTypeSelection',
            project: { name: 'Project #2', task_type: taskType },
        });

        server.use(
            http.get('/api/projects', () => HttpResponse.json([mockedProject])),
            http.get('/api/staged_datasets/{staged_dataset_id}', () => {
                return HttpResponse.json({
                    id: mockedStagedDatasetId,
                    format,
                    size: 123,
                    metadata: {
                        labels: [],
                        num_images: 10,
                        num_frames: 8,
                        num_annotations: 0,
                        num_videos: 1,
                        num_annotated_frames: 5,
                        num_annotated_images: 9,
                        annotation_type: annotationType,
                    },
                    compressed: true,
                    ready_for_export: false,
                    ready_for_import: true,
                });
            })
        );

        render(<ImportTaskSelection stagedDatasetId={mockedStagedDatasetId} />);
    };

    it('shows all three task type options for bounding_box annotations', async () => {
        renderApp({ format: 'geti', taskType: 'detection', annotationType: 'bounding_box' });

        const listButton = await screen.findByRole('button', { name: /Task type/i });
        expect(listButton).toBeVisible();

        await userEvent.click(listButton);
        const container = await screen.findByRole('listbox');

        expect(within(container).getByText('Object detection (Recommended)')).toBeVisible();
        expect(within(container).getByText('Classification')).toBeVisible();
        expect(within(container).getByText('Instance segmentation')).toBeVisible();
    });

    it('shows all three task type options for polygon annotations', async () => {
        renderApp({ format: 'geti', taskType: 'instance_segmentation', annotationType: 'polygon' });

        const listButton = await screen.findByRole('button', { name: /Task type/i });
        expect(listButton).toBeVisible();

        await userEvent.click(listButton);
        const container = await screen.findByRole('listbox');

        expect(within(container).getByText('Object detection')).toBeVisible();
        expect(within(container).getByText('Classification')).toBeVisible();
        expect(within(container).getByText('Instance segmentation (Recommended)')).toBeVisible();
    });

    it('shows only Classification option for label annotations', async () => {
        renderApp({ format: 'geti', taskType: 'classification', annotationType: 'label' });

        const listButton = await screen.findByRole('button', { name: /Task type/i });
        expect(listButton).toBeVisible();

        await userEvent.click(listButton);
        const container = await screen.findByRole('listbox');

        expect(within(container).getByText('Classification (Recommended)')).toBeVisible();
        expect(within(container).queryByText('Object detection')).not.toBeInTheDocument();
        expect(within(container).queryByText('Instance segmentation')).not.toBeInTheDocument();
    });

    it('does not show recommended task types for non-geti format', async () => {
        renderApp({ format: 'coco', annotationType: 'bounding_box' });

        expect(await screen.findByRole('button', { name: /Task type/i })).toHaveTextContent(/Select task/i);
    });

    it('does not show recommended message when there is no recommended task type', async () => {
        renderApp({ format: 'coco', annotationType: 'bounding_box' });

        await screen.findByRole('button', { name: /Task type/i });

        expect(
            screen.queryByText(/The recommended choice is based on the type of the annotations/i)
        ).not.toBeInTheDocument();
    });

    it('shows recommended message when there is a recommended task type', async () => {
        renderApp({ format: 'geti', taskType: 'detection', annotationType: 'bounding_box' });

        expect(
            await screen.findByText(/The recommended choice is based on the type of the annotations/i)
        ).toBeVisible();
    });

    it('shows an error when the project name already exists', async () => {
        renderApp({ format: 'geti', taskType: 'detection', annotationType: 'bounding_box' });

        const projectNameInput = await screen.findByLabelText('Project name');

        await userEvent.clear(projectNameInput);
        await userEvent.type(projectNameInput, `${mockedProject.name}   `);

        expect(await screen.findByText('That project name already exists')).toBeVisible();
    });
});
