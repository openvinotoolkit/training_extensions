// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { isEqual } from 'lodash-es';

import { useTrainModelMutation } from '../api/use-train-model-mutation';
import { useUpdateTrainingConfigurationMutation } from '../api/use-update-training-configuration-mutation';
import { useTrainModelState } from '../train-model-provider.component';
import { getTrainingConfigurationUpdatePayload } from './utils';

export const useTrainModel = () => {
    const projectIdentifier = useProjectIdentifier();
    const trainModelMutation = useTrainModelMutation();
    const updateTrainingConfigurationMutation = useUpdateTrainingConfigurationMutation();

    const {
        selectedTrainingDevice,
        selectedDatasetRevisionId,
        resolvedModelArchitectureId,
        isAdvancedSettingsMode,
        trainingConfiguration,
        selectedModelRevisionId,
        defaultTrainingConfiguration,
        datasetRevisions,
        modelRevisions,
    } = useTrainModelState();

    /**
     * Triggers model training with the following workflow:
     *
     * 1. Direct training — if the user is NOT in advanced settings mode, or IS in advanced
     *    mode but has not changed anything from the defaults, training is started immediately
     *    without touching the configuration.
     *
     * 2. Configuration update + training — if the user IS in advanced mode AND has modified
     *    the configuration:
     *      a. Persist the updated configuration first.
     *      b. On success, start the training job.
     *      c. If the training job itself fails, roll the configuration back to the default
     *         values so the project is not left in an inconsistent state.
     */
    const trainModel = ({ onSuccess }: { onSuccess: () => void }) => {
        if (resolvedModelArchitectureId === null || selectedTrainingDevice === null) {
            return;
        }

        const datasetRevisionId =
            datasetRevisions.find((revision) => revision.id === selectedDatasetRevisionId)?.value ?? null;
        const parentModelRevisionId =
            modelRevisions.find((revision) => revision.id === selectedModelRevisionId)?.value ?? null;

        const trainModelMutationBody = {
            body: {
                project_id: projectIdentifier,
                job_type: 'train',
                parameters: {
                    device: selectedTrainingDevice,
                    model_architecture_id: resolvedModelArchitectureId,
                    parent_model_revision_id: parentModelRevisionId,
                    dataset_revision_id: datasetRevisionId,
                },
            },
        } as const;

        const hasConfigurationChanged = !isEqual(defaultTrainingConfiguration, trainingConfiguration);

        if (!isAdvancedSettingsMode || !hasConfigurationChanged) {
            trainModelMutation.mutate(trainModelMutationBody, { onSuccess });

            return;
        }

        const configurationParams = {
            params: {
                path: {
                    project_id: projectIdentifier,
                },
                query: {
                    model_architecture_id: resolvedModelArchitectureId,
                },
            },
        };

        updateTrainingConfigurationMutation.mutate(
            {
                ...configurationParams,
                body: getTrainingConfigurationUpdatePayload(trainingConfiguration),
            },
            {
                onSuccess: () => {
                    trainModelMutation.mutate(trainModelMutationBody, {
                        onSuccess,
                        onError: () => {
                            updateTrainingConfigurationMutation.mutate({
                                ...configurationParams,
                                body: getTrainingConfigurationUpdatePayload(defaultTrainingConfiguration),
                            });
                        },
                    });
                },
            }
        );
    };

    return { trainModel, isPending: trainModelMutation.isPending || updateTrainingConfigurationMutation.isPending };
};
