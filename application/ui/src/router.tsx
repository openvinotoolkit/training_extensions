// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createBrowserRouter, Navigate, Outlet } from 'react-router';

import { paths } from './constants/paths';
import { ImportDatasetDialogStateProvider } from './features/dataset/providers/export-import-dataset-dialog-provider.component';
import { SelectedDataProvider } from './features/dataset/providers/selected-data-provider.component';
import { WebRTCConnectionProvider } from './features/inference/stream/web-rtc-connection-provider';
import { JobsButton } from './features/jobs/jobs-button.component';
import { ProjectList } from './features/project/list/project-list.component';
import { ImportDatasetDialogProvider } from './features/project/providers/import-dataset-dialog-provider.component';
import { useProjects } from './hooks/api/project.hook';
import { Layout } from './layout';
import { useTerminateAnnotatorWorkersOnUnmount } from './modules/annotator/tools/terminate-annotator-workers-on-unmount.hook';
import { Dataset } from './routes/dataset/dataset.component';
import { ErrorPage } from './routes/error-page/error-page';
import { Inference } from './routes/inference/inference';
import { Models } from './routes/models/models';
import { CreateProject } from './routes/project/create-project';
import { RootLayout } from './routes/root/root';

const Redirect = () => {
    let path = paths.project.index({});

    const { data: projects } = useProjects();

    // No projects -> fall through to the default path (projects list), letting users create or import a project

    // Only 1 project -> Redirect to the dataset page
    if (projects.length === 1) {
        const projectId = projects[0].id;

        if (projectId) {
            path = paths.project.dataset.index({ projectId });
        } else {
            path = paths.project.new({});
        }
    } else if (projects.length > 1) {
        // More than 1 project -> Redirect to the active one
        // And if none are active, fall through to the default path (projects list)
        const projectWithActivePipeline = projects.find((project) => Boolean(project.active_pipeline));

        if (projectWithActivePipeline) {
            path = paths.project.dataset.index({ projectId: projectWithActivePipeline.id });
        }
    }

    return <Navigate to={path} replace />;
};

// Owns the lifetime of the annotator-tool web workers (SAM/SSIM/scissors): they boot lazily inside
// the annotator and are terminated here, freeing the ~100 MB WASM heap when the user leaves the dataset section.
const DatasetSection = () => {
    useTerminateAnnotatorWorkersOnUnmount();

    return (
        <ImportDatasetDialogStateProvider>
            <SelectedDataProvider>
                <Outlet />
            </SelectedDataProvider>
        </ImportDatasetDialogStateProvider>
    );
};

export const router = createBrowserRouter([
    {
        path: paths.root.pattern,
        element: <RootLayout />,
        errorElement: <ErrorPage />,
        children: [
            {
                index: true,
                element: <Redirect />,
            },
            {
                path: paths.project.index.pattern,
                element: (
                    <ImportDatasetDialogProvider>
                        <ProjectList jobsButton={<JobsButton />} />
                    </ImportDatasetDialogProvider>
                ),
            },
            {
                path: paths.project.new.pattern,
                element: <CreateProject />,
            },
            {
                path: paths.project.details.pattern,
                element: <Layout />,
                children: [
                    {
                        index: true,
                        path: paths.project.inference.pattern,
                        element: (
                            <WebRTCConnectionProvider>
                                <Inference />
                            </WebRTCConnectionProvider>
                        ),
                    },
                    {
                        path: paths.project.dataset.index.pattern,
                        element: <DatasetSection />,
                        children: [
                            {
                                index: true,
                                element: <Dataset />,
                            },
                            {
                                path: paths.project.dataset.item.index.pattern,
                                element: <Dataset />,
                            },
                            {
                                path: paths.project.dataset.item.frame.pattern,
                                element: <Dataset />,
                            },
                        ],
                    },
                    {
                        path: paths.project.models.pattern,
                        element: <Models />,
                    },
                ],
            },
            {
                path: '*',
                element: <Redirect />,
            },
        ],
    },
]);
