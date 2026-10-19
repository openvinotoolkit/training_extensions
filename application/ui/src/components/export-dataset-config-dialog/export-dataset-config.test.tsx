// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Project } from '@/api/types';
import { fireEvent, screen, waitFor } from '@testing-library/react';
import { getMockedJob } from 'mocks/mock-job';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { getMockedDatasetStatistics } from '../../../mocks/mock-dataset-item';
import { getMockedProject } from '../../../mocks/mock-project';
import { http } from '../../api/utils';
import { server } from '../../msw-node-setup';
import { ExportDatasetConfig } from './export-dataset-config.component';

describe('ExportDatasetConfig', () => {
    const mockDialogState = {
        isOpen: true,
        open: rs.fn(),
        close: rs.fn(),
        toggle: rs.fn(),
        setOpen: rs.fn(),
        point: { x: 0, y: 0 },
        setPoint: rs.fn(),
    };

    const VIDEO_WARNING = /Exporting videos is not supported by this dataset format/i;
    const EMPTY_LABEL_WARNING_NO_OBJECT = /does not support empty labels.*"No object"/i;
    const EMPTY_LABEL_WARNING_NO_LABEL = /does not support empty labels.*"No label"/i;
    const COCO_WARNING = /The exported dataset won't include any information about the subset assigned to each media/i;

    const renderApp = (project: Project, { videos = 2, emptyLabelInstances = 3 } = {}) => {
        server.use(
            http.get('/api/projects/{project_id}', () => {
                return HttpResponse.json(project);
            }),
            http.get('/api/projects/{project_id}/dataset/statistics', () => {
                const statistics = getMockedDatasetStatistics();

                return HttpResponse.json({
                    ...statistics,
                    media_counts: { ...statistics.media_counts, videos },
                    annotations_counts: {
                        ...statistics.annotations_counts,
                        instances_per_label: [{ label_id: null, instances: emptyLabelInstances }],
                    },
                });
            }),
            http.get('/api/projects/{project_id}/dataset/items', () => {
                return HttpResponse.json({
                    pagination: { total: 0, offset: 0, limit: 0, count: 0 },
                    items: [],
                });
            })
        );

        render(<ExportDatasetConfig dialogState={mockDialogState} datasetId={null} statistics={undefined} />);
    };

    it('shows only GETI export option for classification task', async () => {
        renderApp(
            getMockedProject({
                task: { exclusive_labels: true, task_type: 'classification' },
            })
        );

        expect(await screen.findByText('Export dataset')).toBeVisible();
        expect(screen.getByRole('radio', { name: 'Geti' })).toBeVisible();
        expect(screen.queryByRole('radio', { name: 'YOLO' })).not.toBeInTheDocument();
        expect(screen.queryByRole('radio', { name: 'COCO' })).not.toBeInTheDocument();
    });

    it('shows GETI and COCO export option for instance_segmentation task', async () => {
        renderApp(
            getMockedProject({
                task: { exclusive_labels: true, task_type: 'instance_segmentation' },
            })
        );

        expect(await screen.findByText('Export dataset')).toBeVisible();
        expect(screen.getByRole('radio', { name: 'Geti' })).toBeVisible();
        expect(screen.queryByRole('radio', { name: 'COCO' })).toBeVisible();
    });

    it('does not show the video export warning when the default Geti format is selected', async () => {
        renderApp(getMockedProject({ task: { exclusive_labels: true, task_type: 'instance_segmentation' } }));

        expect(await screen.findByText('Export dataset')).toBeVisible();
        expect(screen.getByRole('radio', { name: 'Geti' })).toBeChecked();
        expect(screen.queryByText(VIDEO_WARNING)).not.toBeInTheDocument();
    });

    it('shows the video export warning when a non-Geti format is selected', async () => {
        renderApp(getMockedProject({ task: { exclusive_labels: true, task_type: 'instance_segmentation' } }));

        fireEvent.click(await screen.findByRole('radio', { name: 'COCO' }));
        expect(screen.getByRole('radio', { name: 'COCO' })).toBeChecked();
        expect(await screen.findByText(VIDEO_WARNING)).toBeVisible();

        fireEvent.click(screen.getByRole('radio', { name: 'Geti' }));
        expect(screen.getByRole('radio', { name: 'Geti' })).toBeChecked();
        expect(screen.queryByText(VIDEO_WARNING)).not.toBeInTheDocument();
    });

    it('shows the empty label warning for a task with empty labels when a non-Geti format is selected', async () => {
        renderApp(getMockedProject({ task: { exclusive_labels: true, task_type: 'detection' } }));

        fireEvent.click(await screen.findByRole('radio', { name: 'COCO' }));
        expect(screen.getByRole('radio', { name: 'COCO' })).toBeChecked();
        expect(await screen.findByText(EMPTY_LABEL_WARNING_NO_OBJECT)).toBeVisible();

        fireEvent.click(screen.getByRole('radio', { name: 'Geti' }));
        expect(screen.getByRole('radio', { name: 'Geti' })).toBeChecked();
        expect(screen.queryByText(EMPTY_LABEL_WARNING_NO_OBJECT)).not.toBeInTheDocument();
    });

    it('does not show the empty label warning for a single-label classification task', async () => {
        renderApp(getMockedProject({ task: { exclusive_labels: true, task_type: 'classification' } }));

        fireEvent.click(await screen.findByRole('radio', { name: 'VOC' }));
        expect(screen.getByRole('radio', { name: 'VOC' })).toBeChecked();

        expect(await screen.findByText(VIDEO_WARNING)).toBeVisible();
        expect(screen.queryByText(EMPTY_LABEL_WARNING_NO_OBJECT)).not.toBeInTheDocument();
        expect(screen.queryByText(EMPTY_LABEL_WARNING_NO_LABEL)).not.toBeInTheDocument();
    });

    it('does not show the video warning when the dataset has no videos', async () => {
        renderApp(getMockedProject({ task: { exclusive_labels: true, task_type: 'detection' } }), { videos: 0 });

        fireEvent.click(await screen.findByRole('radio', { name: 'COCO' }));
        expect(await screen.findByText(EMPTY_LABEL_WARNING_NO_OBJECT)).toBeVisible();
        expect(screen.queryByText(VIDEO_WARNING)).not.toBeInTheDocument();
    });

    it('does not show the empty label warning when the dataset has no empty labels', async () => {
        renderApp(getMockedProject({ task: { exclusive_labels: true, task_type: 'detection' } }), {
            emptyLabelInstances: 0,
        });

        fireEvent.click(await screen.findByRole('radio', { name: 'COCO' }));
        expect(await screen.findByText(VIDEO_WARNING)).toBeVisible();
        expect(screen.queryByText(EMPTY_LABEL_WARNING_NO_OBJECT)).not.toBeInTheDocument();
    });

    it('does not show any warning when the dataset has neither videos nor empty labels', async () => {
        renderApp(getMockedProject({ task: { exclusive_labels: true, task_type: 'detection' } }), {
            videos: 0,
            emptyLabelInstances: 0,
        });

        fireEvent.click(await screen.findByRole('radio', { name: 'COCO' }));
        expect(screen.getByRole('radio', { name: 'COCO' })).toBeChecked();

        await expect(screen.findByText(/please use the Geti export format/i)).rejects.toThrow();
    });

    it('shows the COCO subset warning', async () => {
        renderApp(getMockedProject({ task: { exclusive_labels: true, task_type: 'detection' } }), { videos: 0 });

        fireEvent.click(await screen.findByRole('radio', { name: 'COCO' }));
        expect(await screen.findByText(COCO_WARNING)).toBeVisible();
    });

    describe('dataset views', () => {
        const renderViewApp = () => {
            let exportRequestBody: Record<string, unknown> | undefined;

            server.use(
                http.get('/api/projects/{project_id}', () => {
                    return HttpResponse.json(
                        getMockedProject({ task: { exclusive_labels: true, task_type: 'detection' } })
                    );
                }),
                http.post('/api/jobs', async ({ request }) => {
                    exportRequestBody = (await request.json()) as Record<string, unknown>;

                    return HttpResponse.json(getMockedJob({ job_id: 'job-1' }));
                })
            );

            render(
                <ExportDatasetConfig
                    dialogState={mockDialogState}
                    datasetId={null}
                    datasetViewId='view-1'
                    datasetViewName='Canada signs'
                    statistics={undefined}
                />
            );

            return () => exportRequestBody;
        };

        it('shows the name of the dataset view being exported', async () => {
            renderViewApp();

            expect(await screen.findByRole('heading', { name: 'Export dataset view' })).toBeVisible();
            expect(screen.getByText('View: Canada signs')).toBeVisible();
        });

        it('sends the dataset view id when submitting the export job', async () => {
            const getExportRequestBody = renderViewApp();

            fireEvent.click(await screen.findByRole('button', { name: 'Export' }));

            await waitFor(() => expect(getExportRequestBody()).toBeDefined());
            expect(getExportRequestBody()).toEqual(
                expect.objectContaining({ dataset_id: null, dataset_view_id: 'view-1' })
            );
        });

        it('does not show a view row when exporting the entire dataset', async () => {
            renderApp(getMockedProject({ task: { exclusive_labels: true, task_type: 'detection' } }));

            expect(await screen.findByText('Export dataset')).toBeVisible();
            expect(screen.queryByText(/^View: /)).not.toBeInTheDocument();
        });
    });
});
