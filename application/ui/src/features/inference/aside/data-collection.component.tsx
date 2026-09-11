// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { useTranslation } from '@/i18n';
import { Divider, Flex, Heading, NumberField, Slider, Switch, Text } from '@geti-ui/ui';
import { usePatchPipeline, usePipeline } from 'hooks/api/pipeline.hook';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

const DEFAULTS = {
    RATE: 12,
    CONFIDENCE_THRESHOLD: 0.5,
    MIN_SAMPLING_INTERVAL: 2.5,
    MAX_DATASET_SIZE: 500,
} as const;

const MIN_FRAME_SAMPLING_VALUE = 0.1;

const isPositiveFiniteNumber = (value: number): boolean => Number.isFinite(value) && value > 0;

export const DataCollection = () => {
    const { t } = useTranslation();
    const projectId = useProjectIdentifier();
    const pipelineQuery = usePipeline();
    const patchPipelineMutation = usePatchPipeline();

    const policies = pipelineQuery.data?.data_collection?.policies ?? [];
    const ratePolicy = policies.find((policy) => policy.type === 'fixed_rate');
    const confidencePolicy = policies.find((policy) => policy.type === 'confidence_threshold');

    const maxDatasetSize = pipelineQuery.data?.data_collection?.max_dataset_size ?? DEFAULTS.MAX_DATASET_SIZE;
    const serverRate = ratePolicy?.rate ?? DEFAULTS.RATE;
    const serverConfidenceThreshold = confidencePolicy?.confidence_threshold ?? DEFAULTS.CONFIDENCE_THRESHOLD;

    const [localRateFrames, setLocalRateFrames] = useState(serverRate);
    const [localRateSeconds, setLocalRateSeconds] = useState(1);
    const [localConfidenceThreshold, setLocalConfidenceThreshold] = useState(serverConfidenceThreshold);

    const isUpdating = patchPipelineMutation.isPending;

    const updateRatePolicy = (frames: number, seconds: number) => {
        if (!isPositiveFiniteNumber(frames) || !isPositiveFiniteNumber(seconds)) {
            return;
        }

        updatePolicies({ rate: frames / seconds });
    };

    const updatePolicies = (updates: {
        maxDatasetSize?: number;
        rateEnabled?: boolean;
        rate?: number;
        confidenceEnabled?: boolean;
        confidenceThreshold?: number;
    }) => {
        const newDataCollectionPolicies = {
            max_dataset_size: updates.maxDatasetSize ?? maxDatasetSize,
            policies: [
                {
                    type: 'fixed_rate' as const,
                    rate: updates.rate ?? serverRate,
                    enabled: updates.rateEnabled ?? ratePolicy?.enabled ?? false,
                },
                {
                    type: 'confidence_threshold' as const,
                    confidence_threshold: updates.confidenceThreshold ?? serverConfidenceThreshold,
                    min_sampling_interval: confidencePolicy?.min_sampling_interval ?? DEFAULTS.MIN_SAMPLING_INTERVAL,
                    enabled: updates.confidenceEnabled ?? confidencePolicy?.enabled ?? false,
                },
            ],
        };

        patchPipelineMutation.mutate(
            {
                params: { path: { project_id: projectId } },
                body: { data_collection: newDataCollectionPolicies },
            },
            {
                onError: () => {
                    setLocalRateFrames(serverRate);
                    setLocalRateSeconds(1);
                },
            }
        );
    };

    return (
        <Flex direction={'column'} minHeight={0} height={'100%'}>
            <Flex direction={'column'} flex={1} UNSAFE_style={{ overflow: 'hidden auto' }}>
                <Heading level={3} margin={0}>
                    {t('inference.collection.maxDatasetSize.title')}
                </Heading>

                <Text marginY={'size-100'}>{t('inference.collection.maxDatasetSize.description')}</Text>

                <NumberField
                    label={t('inference.collection.maxDatasetSize.sizeLabel')}
                    width={'100%'}
                    minValue={1}
                    step={1}
                    value={maxDatasetSize}
                    onChange={(nextMaxDatasetSize) => {
                        updatePolicies({ maxDatasetSize: nextMaxDatasetSize });
                    }}
                    isDisabled={isUpdating}
                />

                <Divider marginY={'size-400'} size={'S'} />

                <Heading level={3} margin={0}>
                    {t('inference.collection.captureRate.title')}
                </Heading>

                <Text marginY={'size-100'}>{t('inference.collection.captureRate.description')}</Text>

                <Switch
                    isEmphasized
                    isSelected={ratePolicy?.enabled ?? false}
                    onChange={(enabled) => updatePolicies({ rateEnabled: enabled })}
                    marginBottom={'size-200'}
                    isDisabled={isUpdating}
                >
                    {t('inference.collection.captureRate.toggleLabel')}
                </Switch>

                <Flex direction='row' gap='size-100' alignItems={'end'} marginBottom={'size-200'}>
                    <NumberField
                        label={t('inference.collection.captureRate.framesLabel')}
                        minValue={MIN_FRAME_SAMPLING_VALUE}
                        step={0.1}
                        value={localRateFrames}
                        onChange={(nextFrames) => {
                            setLocalRateFrames(nextFrames);
                            updateRatePolicy(nextFrames, localRateSeconds);
                        }}
                        isDisabled={!ratePolicy?.enabled || isUpdating}
                    />
                    <Text>{t('inference.collection.captureRate.every')}</Text>
                    <NumberField
                        label={t('inference.collection.captureRate.secondsLabel')}
                        minValue={1}
                        step={1}
                        value={localRateSeconds}
                        onChange={(nextSeconds) => {
                            setLocalRateSeconds(nextSeconds);
                            updateRatePolicy(localRateFrames, nextSeconds);
                        }}
                        isDisabled={!ratePolicy?.enabled || isUpdating}
                    />
                </Flex>

                <Divider marginY={'size-400'} size={'S'} />

                <Heading level={3} margin={0}>
                    {t('inference.collection.confidenceThreshold.title')}
                </Heading>

                <Text marginY={'size-100'}>{t('inference.collection.confidenceThreshold.description')}</Text>

                <Switch
                    isEmphasized
                    isSelected={confidencePolicy?.enabled ?? false}
                    onChange={(enabled) => updatePolicies({ confidenceEnabled: enabled })}
                    isDisabled={isUpdating}
                >
                    {t('inference.collection.confidenceThreshold.toggleLabel')}
                </Switch>

                <Slider
                    isFilled
                    step={0.01}
                    minValue={0}
                    maxValue={1}
                    value={localConfidenceThreshold}
                    onChange={setLocalConfidenceThreshold}
                    onChangeEnd={(confidenceThreshold) => updatePolicies({ confidenceThreshold })}
                    marginY={'size-200'}
                    label={t('inference.collection.confidenceThreshold.sliderLabel')}
                    isDisabled={!confidencePolicy?.enabled || isUpdating}
                />
            </Flex>
        </Flex>
    );
};
