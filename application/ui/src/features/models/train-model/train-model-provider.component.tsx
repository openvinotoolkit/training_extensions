// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, Dispatch, ReactNode, SetStateAction, use, useMemo, useState } from 'react';

import type {
    Model,
    ModelArchitectureWithPerformanceCategory,
    TrainingConfiguration,
    TrainingDevice,
} from '@/api/types';
import { useTranslation } from '@/i18n';
import { useDatasetViewsQuery } from 'hooks/api/dataset-views.hook';
import { useGetSuccessfulModels } from 'hooks/api/use-get-models.hook';
import { useDatasetViewId } from 'hooks/use-dataset-view-id.hook';
import { useGetDatasetRevisions } from 'hooks/use-get-dataset-revisions.hook';

import { useGetTaskModelArchitectures } from '../hooks/api/use-get-model-architectures.hook';
import { useGetTrainingDevices } from './api/use-get-training-devices';
import { useTimmModelSelection, type TimmModelSelection } from './hooks/use-timm-model-selection';
import { useTrainingConfiguration } from './hooks/use-training-configuration';
import { getDefaultTrainingDevice } from './select-training-device/utils';
import { isTimmModelArchitecture } from './timm-model-configuration/utils';

/** Where the training data comes from: the live dataset, one of its views, or a frozen revision. */
export type DatasetSourceKind = 'current' | 'view' | 'revision';

export type DatasetSource = {
    /** Unique key of the entry in the dataset picker. */
    id: string;
    name: string;
    kind: DatasetSourceKind;
    /** Identifier sent to the backend; null when training on the current dataset. */
    value: string | null;
};

export const CURRENT_DATASET_SOURCE_ID = 'use-current-dataset';

export const getViewSourceId = (datasetViewId: string) => `view-${datasetViewId}`;

export const getRevisionSourceId = (datasetRevisionId: string) => `revision-${datasetRevisionId}`;

type ModelRevisionWithValue = Pick<Model, 'id' | 'name' | 'architecture'> & { value: string | null };

export type TrainModelContextProps = TimmModelSelection & {
    modelArchitectures: ModelArchitectureWithPerformanceCategory[];

    selectedModelArchitectureId: string | null;
    onSelectModelArchitectureId: (id: string | null) => void;

    /**
     * Architecture id that training is actually started with. Equal to the selected
     * architecture id, except for the synthetic timm card, where it resolves to the
     * concrete timm backbone id (or `null` while no backbone is resolved yet).
     */
    resolvedModelArchitectureId: string | null;

    trainingDevices: TrainingDevice[];
    selectedTrainingDevice: string | null;
    onSelectTrainingDevice: (deviceKey: string | null) => void;

    datasetSources: DatasetSource[];
    selectedDatasetSourceId: string | null;
    onSelectDatasetSourceId: (datasetSourceId: string | null) => void;

    modelRevisions: ModelRevisionWithValue[];
    selectedModelRevisionId: string | null;
    onSelectModelRevisionId: (modelRevisionId: string | null) => void;

    isAdvancedSettingsMode: boolean;
    onToggleAdvancedSettingsMode: (isAdvancedSettingsMode: boolean) => void;

    trainingConfiguration: TrainingConfiguration | undefined;
    defaultTrainingConfiguration: TrainingConfiguration | undefined;
    onTrainingConfigurationChange: Dispatch<SetStateAction<TrainingConfiguration | undefined>>;

    showMoreModelArchitectures: boolean;
    onToggleShowMoreModelArchitectures: (showMore: boolean) => void;
};

const TrainModelContext = createContext<TrainModelContextProps | null>(null);

type TrainModelProviderProps = {
    children: ReactNode;
};

const useDatasetSources = (): DatasetSource[] => {
    const { t } = useTranslation();
    const { data: datasetRevisions } = useGetDatasetRevisions();
    const { data: datasetViews } = useDatasetViewsQuery();

    return useMemo(
        () => [
            {
                id: CURRENT_DATASET_SOURCE_ID,
                name: t('models.training.setup.selectDataset.useEntireDataset'),
                kind: 'current' as const,
                value: null,
            },
            ...(datasetViews?.map(({ id, name }) => ({
                id: getViewSourceId(String(id)),
                name,
                kind: 'view' as const,
                value: String(id),
            })) ?? []),
            ...(datasetRevisions?.map(({ id, name }) => ({
                id: getRevisionSourceId(String(id)),
                name,
                kind: 'revision' as const,
                value: String(id),
            })) ?? []),
        ],
        [t, datasetViews, datasetRevisions]
    );
};

const DEFAULT_PRE_TRAINED_WEIGHTS = 'default-pre-trained-weights';
const useModelRevisions = () => {
    const { t } = useTranslation();
    const { data: models } = useGetSuccessfulModels();

    return {
        modelRevisions: [
            {
                id: DEFAULT_PRE_TRAINED_WEIGHTS,
                name: t('models.training.setup.selectModel.defaultPretrainedWeights'),
                architecture: '',
                value: null,
            },
            ...(models?.map(({ id, name, architecture }) => ({ id, name, architecture, value: String(id) })) ?? []),
        ],
    };
};

const getModelRevisionsForArchitecture = (
    modelRevisions: ModelRevisionWithValue[],
    architectureId: string | null
): ModelRevisionWithValue[] => {
    return modelRevisions.filter((modelRevision) => {
        if (modelRevision.id === DEFAULT_PRE_TRAINED_WEIGHTS) {
            return true;
        }

        return modelRevision.architecture === architectureId;
    });
};

const getDefaultModelRevisionIdForArchitecture = (
    modelRevisions: ModelRevisionWithValue[],
    architectureId: string | null
): string | null => {
    const revisionsForArchitecture = getModelRevisionsForArchitecture(modelRevisions, architectureId);
    const firstRevision = revisionsForArchitecture.find(({ id }) => id !== DEFAULT_PRE_TRAINED_WEIGHTS);

    return firstRevision?.id ?? revisionsForArchitecture.at(0)?.id ?? null;
};

export const createTrainingDeviceKey = (trainingDevice: TrainingDevice): string => {
    if (trainingDevice.index == null) {
        return trainingDevice.type;
    }

    return `${trainingDevice.type}-${trainingDevice.index}`;
};

export const TrainModelProvider = ({ children }: TrainModelProviderProps) => {
    const { modelArchitectures } = useGetTaskModelArchitectures();
    const { data: trainingDevices } = useGetTrainingDevices();
    const datasetSources = useDatasetSources();
    const { modelRevisions: allModelRevisions } = useModelRevisions();
    const [datasetViewId] = useDatasetViewId();

    const [selectedModelArchitectureId, setSelectedModelArchitectureId] = useState<string | null>(null);

    const isTimmArchitectureSelected = isTimmModelArchitecture(selectedModelArchitectureId);
    const timmModelSelection = useTimmModelSelection(isTimmArchitectureSelected);

    const resolvedModelArchitectureId = isTimmArchitectureSelected
        ? (timmModelSelection.timmModelArchitecture?.id ?? null)
        : selectedModelArchitectureId;

    const [selectedTrainingDevice, setSelectedTrainingDevice] = useState<string | null>(() => {
        const defaultDevice = getDefaultTrainingDevice(trainingDevices);
        return defaultDevice ? createTrainingDeviceKey(defaultDevice) : null;
    });
    // Training started from the Dataset screen defaults to the view that is currently open there.
    const [selectedDatasetSourceId, setSelectedDatasetSourceId] = useState<string | null>(() => {
        const viewSourceId = datasetViewId === null ? null : getViewSourceId(datasetViewId);

        return viewSourceId !== null && datasetSources.some(({ id }) => id === viewSourceId)
            ? viewSourceId
            : CURRENT_DATASET_SOURCE_ID;
    });
    const [modelRevisionId, setModelRevisionId] = useState<string | null>(() =>
        getDefaultModelRevisionIdForArchitecture(allModelRevisions, selectedModelArchitectureId)
    );

    const [isAdvancedSettingsMode, setIsAdvancedSettingsMode] = useState<boolean>(false);

    const modelRevisions = getModelRevisionsForArchitecture(allModelRevisions, resolvedModelArchitectureId);

    // The resolved architecture also changes while the timm card stays selected, which can invalidate the selection
    const selectedModelRevisionId = modelRevisions.some(({ id }) => id === modelRevisionId)
        ? modelRevisionId
        : getDefaultModelRevisionIdForArchitecture(allModelRevisions, resolvedModelArchitectureId);

    const selectedModelRevision = modelRevisions.find((modelRevision) => modelRevision.id === selectedModelRevisionId);

    const [trainingConfiguration, setTrainingConfiguration, defaultTrainingConfiguration] = useTrainingConfiguration({
        modelArchitectureId: resolvedModelArchitectureId,
        modelRevisionId: selectedModelRevision?.value ?? null,
    });

    const [showMoreModelArchitectures, setShowMoreModelArchitectures] = useState<boolean>(false);

    const onSelectModelArchitectureId = (modelArchitectureId: string | null) => {
        setSelectedModelArchitectureId(modelArchitectureId);
        setModelRevisionId(getDefaultModelRevisionIdForArchitecture(allModelRevisions, modelArchitectureId));
    };

    return (
        <TrainModelContext
            value={{
                ...timmModelSelection,

                modelArchitectures,

                selectedModelArchitectureId,
                onSelectModelArchitectureId,
                resolvedModelArchitectureId,

                trainingDevices,
                selectedTrainingDevice,
                onSelectTrainingDevice: setSelectedTrainingDevice,

                datasetSources,
                selectedDatasetSourceId,
                onSelectDatasetSourceId: setSelectedDatasetSourceId,

                modelRevisions,
                selectedModelRevisionId,
                onSelectModelRevisionId: setModelRevisionId,

                isAdvancedSettingsMode,
                onToggleAdvancedSettingsMode: setIsAdvancedSettingsMode,

                showMoreModelArchitectures,
                onToggleShowMoreModelArchitectures: setShowMoreModelArchitectures,

                trainingConfiguration,
                defaultTrainingConfiguration,
                onTrainingConfigurationChange: setTrainingConfiguration,
            }}
        >
            {children}
        </TrainModelContext>
    );
};

export const useTrainModelState = () => {
    const context = use(TrainModelContext);

    if (context === null) {
        throw new Error('useTrainModel must be used within a TrainModelProvider');
    }

    return context;
};
