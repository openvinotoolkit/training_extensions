// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { ConfigurableParameter, ConfigurableParameterGroup } from '@/api/types';
import { act, waitFor } from '@testing-library/react';
import { HttpResponse } from 'msw';
import { renderHook } from 'test-utils/render';
import { vi } from 'vitest';

import { getMockedJob } from '../../../../../mocks/mock-job';
import { http } from '../../../../api/utils';
import { server } from '../../../../msw-node-setup';
import { findGroupByKey } from '../../model-listing/model-training-parameters/utils';
import { deepReplaceParameters } from '../advanced-settings/utils';
import { TrainModelContextProps } from '../train-model-provider.component';
import { mockedTrainingConfiguration } from './mocks';
import { useTrainModel } from './use-train-model';
import { getTrainingConfigurationUpdatePayload } from './utils';

const DEFAULT_STATE: Partial<TrainModelContextProps> = {
    selectedModelArchitectureId: 'arch-1',
    resolvedModelArchitectureId: 'arch-1',
    selectedTrainingDevice: 'cpu',
    selectedDatasetRevisionId: 'use-current-dataset-revision',
    selectedModelRevisionId: 'default-pre-trained-weights',
    isAdvancedSettingsMode: false,
    trainingConfiguration: mockedTrainingConfiguration,
    defaultTrainingConfiguration: mockedTrainingConfiguration,
    datasetRevisions: [{ id: 'use-current-dataset-revision', name: 'Use current dataset', value: null }],
    modelRevisions: [
        { id: 'default-pre-trained-weights', name: 'Default pre-trained weights', architecture: '', value: null },
    ],
};

const mockTrainModelState = vi.hoisted(() => vi.fn(() => DEFAULT_STATE));

vi.mock('../train-model-provider.component', async () => {
    const actualImport = await vi.importActual('../train-model-provider.component');
    return {
        ...actualImport,
        useTrainModelState: mockTrainModelState,
    };
});

vi.mock('hooks/use-project-identifier.hook', () => ({
    useProjectIdentifier: () => 'project-123',
}));

const mockedJob = getMockedJob({ job_id: 'job-abc' });

const registerSuccessfulTrainHandler = (trainSpy: () => void = vi.fn<() => void>()) => {
    server.use(
        http.post('/api/jobs', () => {
            trainSpy();
            return HttpResponse.json(mockedJob, { status: 201 });
        })
    );
};

const registerFailingTrainHandler = () => {
    // @ts-expect-error: We only care about mocking detail property.
    server.use(http.post('/api/jobs', () => HttpResponse.json({ detail: 'Training failed' }, { status: 500 })));
};

const registerSuccessfulConfigUpdateHandler = () => {
    server.use(
        http.patch('/api/projects/{project_id}/training_configuration', () => HttpResponse.json({}, { status: 200 }))
    );
};

const registerFailingConfigUpdateHandler = () => {
    server.use(
        http.patch('/api/projects/{project_id}/training_configuration', () =>
            // @ts-expect-error: We only care about mocking detail property.
            HttpResponse.json({ detail: 'Config update failed' }, { status: 500 })
        )
    );
};

describe('useTrainModel', () => {
    beforeEach(() => {
        mockTrainModelState.mockReturnValue(DEFAULT_STATE);
    });

    describe('guard clause', () => {
        it('does not call any API when model architecture is not selected', async () => {
            mockTrainModelState.mockReturnValue({ ...DEFAULT_STATE, resolvedModelArchitectureId: null });

            const trainSpy = vi.fn();
            registerSuccessfulTrainHandler(trainSpy);

            const { result } = renderHook(() => useTrainModel());

            act(() => {
                result.current.trainModel({ onSuccess: vi.fn() });
            });

            // Give the async path a chance to fire — it should not.
            await new Promise((resolve) => setTimeout(resolve, 50));
            expect(trainSpy).not.toHaveBeenCalled();
        });

        it('does not call any API when training device is not selected', async () => {
            mockTrainModelState.mockReturnValue({ ...DEFAULT_STATE, selectedTrainingDevice: null });

            const trainSpy = vi.fn();
            registerSuccessfulTrainHandler(trainSpy);

            const { result } = renderHook(() => useTrainModel());

            act(() => {
                result.current.trainModel({ onSuccess: vi.fn() });
            });

            await new Promise((resolve) => setTimeout(resolve, 50));
            expect(trainSpy).not.toHaveBeenCalled();
        });
    });

    describe('direct training', () => {
        it('starts training immediately when not in advanced settings mode', async () => {
            const trainSpy = vi.fn();
            registerSuccessfulTrainHandler(trainSpy);

            const onSuccess = vi.fn();
            const { result } = renderHook(() => useTrainModel());

            act(() => {
                result.current.trainModel({ onSuccess });
            });

            await waitFor(() => {
                expect(trainSpy).toHaveBeenCalled();
                expect(onSuccess).toHaveBeenCalled();
            });
        });

        it('starts training immediately when in advanced mode but configuration is unchanged', async () => {
            mockTrainModelState.mockReturnValue({
                ...DEFAULT_STATE,
                isAdvancedSettingsMode: true,
                trainingConfiguration: mockedTrainingConfiguration,
                defaultTrainingConfiguration: mockedTrainingConfiguration,
            });

            const trainSpy = vi.fn();
            registerSuccessfulTrainHandler(trainSpy);

            const onSuccess = vi.fn();
            const { result } = renderHook(() => useTrainModel());

            act(() => {
                result.current.trainModel({ onSuccess });
            });

            await waitFor(() => {
                expect(trainSpy).toHaveBeenCalled();
                expect(onSuccess).toHaveBeenCalled();
            });
        });

        it('passes the correct job payload to the train mutation', async () => {
            const datasetRevisionId = 'dataset-rev-1';
            const modelRevisionId = 'model-rev-1';

            mockTrainModelState.mockReturnValue({
                ...DEFAULT_STATE,
                selectedDatasetRevisionId: 'ds-entry-1',
                selectedModelRevisionId: 'model-entry-1',
                datasetRevisions: [
                    { id: 'use-current-dataset-revision', name: 'Use current dataset', value: null },
                    { id: 'ds-entry-1', name: 'Rev 1', value: datasetRevisionId },
                ],
                modelRevisions: [
                    {
                        id: 'default-pre-trained-weights',
                        name: 'Default pre-trained weights',
                        architecture: '',
                        value: null,
                    },
                    { id: 'model-entry-1', name: 'Model Rev 1', architecture: 'arch-1', value: modelRevisionId },
                ],
            });

            let capturedBody: unknown;
            server.use(
                http.post('/api/jobs', async ({ request }) => {
                    capturedBody = await request.json();
                    return HttpResponse.json(mockedJob, { status: 201 });
                })
            );

            const { result } = renderHook(() => useTrainModel());

            act(() => {
                result.current.trainModel({ onSuccess: vi.fn() });
            });

            await waitFor(() => expect(capturedBody).toBeDefined());

            expect(capturedBody).toMatchObject({
                project_id: 'project-123',
                job_type: 'train',
                parameters: {
                    device: 'cpu',
                    model_architecture_id: 'arch-1',
                    parent_model_revision_id: modelRevisionId,
                    dataset_revision_id: datasetRevisionId,
                },
            });
        });

        it('uses null for dataset/model revision when the Use current dataset / default pre-trained weights entries are selected', async () => {
            let capturedBody: unknown;
            server.use(
                http.post('/api/jobs', async ({ request }) => {
                    capturedBody = await request.json();
                    return HttpResponse.json(mockedJob, { status: 201 });
                })
            );

            const { result } = renderHook(() => useTrainModel());

            act(() => {
                result.current.trainModel({ onSuccess: vi.fn() });
            });

            await waitFor(() => expect(capturedBody).toBeDefined());

            expect(capturedBody).toMatchObject({
                parameters: {
                    parent_model_revision_id: null,
                    dataset_revision_id: null,
                },
            });
        });
    });

    describe('configuration update + training', () => {
        const datasetPreparationGroup = findGroupByKey(
            mockedTrainingConfiguration.parameters,
            'dataset_preparation'
        ) as ConfigurableParameterGroup;
        const subsetSplit = findGroupByKey(datasetPreparationGroup?.parameters, 'subset_split')
            ?.parameters as ConfigurableParameter[];
        const updatedSubsetSplitTraining = { ...subsetSplit[0], value: 90 } as ConfigurableParameter;

        const changedConfiguration = {
            parameters: deepReplaceParameters(
                mockedTrainingConfiguration.parameters,
                [updatedSubsetSplitTraining],
                ['dataset_preparation', 'subset_split']
            ),
        };

        beforeEach(() => {
            mockTrainModelState.mockReturnValue({
                ...DEFAULT_STATE,
                isAdvancedSettingsMode: true,
                trainingConfiguration: changedConfiguration,
                defaultTrainingConfiguration: mockedTrainingConfiguration,
            });
        });

        it('updates configuration before starting the training job', async () => {
            const callOrder: string[] = [];

            server.use(
                http.patch('/api/projects/{project_id}/training_configuration', () => {
                    callOrder.push('config-update');
                    return HttpResponse.json({}, { status: 200 });
                }),
                http.post('/api/jobs', () => {
                    callOrder.push('train');
                    return HttpResponse.json(mockedJob, { status: 201 });
                })
            );

            const { result } = renderHook(() => useTrainModel());

            act(() => {
                result.current.trainModel({ onSuccess: vi.fn() });
            });

            await waitFor(() => expect(callOrder).toEqual(['config-update', 'train']));
        });

        it('calls onSuccess after a successful config update and training', async () => {
            registerSuccessfulConfigUpdateHandler();
            registerSuccessfulTrainHandler();

            const onSuccess = vi.fn();
            const { result } = renderHook(() => useTrainModel());

            act(() => {
                result.current.trainModel({ onSuccess });
            });

            await waitFor(() => expect(onSuccess).toHaveBeenCalledTimes(1));
        });

        it('does not call onSuccess when the config update fails', async () => {
            registerFailingConfigUpdateHandler();

            const onSuccess = vi.fn();
            const { result } = renderHook(() => useTrainModel());

            act(() => {
                result.current.trainModel({ onSuccess });
            });

            await new Promise((resolve) => setTimeout(resolve, 100));
            expect(onSuccess).not.toHaveBeenCalled();
        });

        it('does not call onSuccess when training fails after a successful config update', async () => {
            registerSuccessfulConfigUpdateHandler();
            registerFailingTrainHandler();

            const onSuccess = vi.fn();
            const { result } = renderHook(() => useTrainModel());

            act(() => {
                result.current.trainModel({ onSuccess });
            });

            await new Promise((resolve) => setTimeout(resolve, 100));
            expect(onSuccess).not.toHaveBeenCalled();
        });

        it('rolls back configuration to defaults when training fails', async () => {
            registerSuccessfulConfigUpdateHandler();
            registerFailingTrainHandler();

            const patchBodies: unknown[] = [];
            server.use(
                http.patch('/api/projects/{project_id}/training_configuration', async ({ request }) => {
                    patchBodies.push(await request.json());
                    return HttpResponse.json({}, { status: 200 });
                }),
                // @ts-expect-error: We only care about mocking detail property.
                http.post('/api/jobs', () => HttpResponse.json({ detail: 'failed' }, { status: 500 }))
            );

            const { result } = renderHook(() => useTrainModel());

            act(() => {
                result.current.trainModel({ onSuccess: vi.fn() });
            });

            // Wait for two PATCH calls: original update + rollback
            await waitFor(() => expect(patchBodies.length).toBe(2));

            // The second PATCH should carry the default configuration payload
            const expectedDefaultPayload = getTrainingConfigurationUpdatePayload(mockedTrainingConfiguration);

            expect(patchBodies[1]).toEqual(expectedDefaultPayload);
        });

        it('sends the changed configuration payload on the first update', async () => {
            let capturedConfigBody: unknown;
            server.use(
                http.patch('/api/projects/{project_id}/training_configuration', async ({ request }) => {
                    capturedConfigBody = await request.json();
                    return HttpResponse.json({}, { status: 200 });
                }),
                http.post('/api/jobs', () => HttpResponse.json(mockedJob, { status: 201 }))
            );

            const { result } = renderHook(() => useTrainModel());

            act(() => {
                result.current.trainModel({ onSuccess: vi.fn() });
            });

            await waitFor(() => expect(capturedConfigBody).toBeDefined());

            expect(capturedConfigBody).toEqual(getTrainingConfigurationUpdatePayload(changedConfiguration));
        });
    });
});
