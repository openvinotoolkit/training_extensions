// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Model, ModelFormat, ModelVariant } from '@/api/types';
import {
    ActionButton,
    Cell,
    Column,
    Content,
    ContextualHelp,
    Flex,
    Heading,
    Row,
    TableBody,
    TableHeader,
    TableView,
    Text,
} from '@geti-ui/ui';
import { DownloadIcon } from '@geti-ui/ui/icons';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { get } from 'lodash-es';
import { useNumberFormatter } from 'react-aria';

import { downloadFile } from '../../../../platform/download-file';
import { formatBytes } from '../../../../shared/util';
import { getModelVariantBinaryFilename, getModelVariantBinaryUrl } from '../utils/utils';
import {
    getBaselineVariant,
    getFp32PytorchVariant,
    getPerformanceColumnName,
    getPrimaryTestingMetricValue,
    getVariantPerformanceValue,
} from '../utils/variant-metrics';
import { ValueWithDelta } from './model-variant-delta.component';

type ModelVariantTableProps = {
    model: Model;
    format: ModelFormat;
};

type ModelVariantPrecisionRendererProps = {
    variant: ModelVariant;
};

const ModelVariantPrecisionRenderer = ({ variant }: ModelVariantPrecisionRendererProps) => {
    const numberFormatter = useNumberFormatter({
        style: 'percent',
        maximumFractionDigits: 1,
    });

    if (variant.quantization_info == null) {
        return <Text>{variant.precision.toUpperCase()}</Text>;
    }

    const quantizationParameters = {
        maxDrop: get(variant.quantization_info, 'max_drop', null),
        maxCalibrationSubsetSize: get(variant.quantization_info, 'max_calibration_subset_size', null),
    };

    const maxAccuracyDrop = quantizationParameters.maxDrop === null ? null : Number(quantizationParameters.maxDrop);
    const calibrationDatasetSize =
        quantizationParameters.maxCalibrationSubsetSize === null
            ? null
            : Number(quantizationParameters.maxCalibrationSubsetSize);

    return (
        <Flex direction={'row'} gap={'size-100'}>
            <Text>{variant.precision.toUpperCase()}</Text>
            {(calibrationDatasetSize || maxAccuracyDrop) && (
                <ContextualHelp variant={'info'} placement={'top'}>
                    <Heading>Quantized with NNCF PTQ</Heading>
                    <Content>
                        <Flex direction={'column'}>
                            {maxAccuracyDrop !== null && (
                                <Text>Max accuracy drop: {numberFormatter.format(maxAccuracyDrop)}</Text>
                            )}
                            {calibrationDatasetSize != null && (
                                <Text>Calibration dataset size: {calibrationDatasetSize}</Text>
                            )}
                        </Flex>
                    </Content>
                </ContextualHelp>
            )}
        </Flex>
    );
};

export const ModelVariantTable = ({ model, format }: ModelVariantTableProps) => {
    const projectId = useProjectIdentifier();

    const allVariants = model.variants ?? [];
    const variants = allVariants.filter((variant) => variant.format === format);
    const baselineVariant = getBaselineVariant(variants);
    const fp32PytorchVariant = getFp32PytorchVariant(allVariants);

    const fp32PytorchMetric = getPrimaryTestingMetricValue(fp32PytorchVariant);
    const performanceColumnName = getPerformanceColumnName(variants, fp32PytorchMetric);
    const baselinePerformanceValue = baselineVariant
        ? getVariantPerformanceValue(baselineVariant, fp32PytorchMetric)
        : undefined;

    const handleDownloadModel = (variant: ModelVariant) => {
        downloadFile(
            getModelVariantBinaryUrl(projectId, model.id, variant.id),
            getModelVariantBinaryFilename(model.id, variant),
            'Model download started'
        );
    };

    return (
        <TableView aria-label={`Model variants for ${model.id}`} overflowMode={'wrap'} density={'compact'}>
            <TableHeader>
                <Column isRowHeader>PRECISION</Column>
                <Column isRowHeader>SIZE</Column>
                <Column isRowHeader>{performanceColumnName}</Column>
                <Column align='end'>
                    <></>
                </Column>
            </TableHeader>
            <TableBody items={variants}>
                {(variant) => {
                    const performanceValue = getVariantPerformanceValue(variant, fp32PytorchMetric);
                    const isBaselineVariant = variant.id === baselineVariant?.id;
                    const areWeightsDeleted = model.files_deleted || variant.files_deleted;

                    return (
                        <Row key={variant.id}>
                            <Cell>
                                <ModelVariantPrecisionRenderer variant={variant} />
                            </Cell>
                            <Cell>
                                <ValueWithDelta
                                    value={variant.weights_size}
                                    baselineValue={baselineVariant?.weights_size}
                                    changeType='size'
                                    displayValue={areWeightsDeleted ? '-' : formatBytes(variant.weights_size)}
                                    showDelta={!isBaselineVariant && !areWeightsDeleted}
                                    precision={variant.precision}
                                />
                            </Cell>
                            <Cell>
                                <ValueWithDelta
                                    value={performanceValue}
                                    baselineValue={baselinePerformanceValue}
                                    displayValue={performanceValue === undefined ? '-' : `${performanceValue}%`}
                                    showDelta={!isBaselineVariant}
                                    precision={variant.precision}
                                />
                            </Cell>
                            <Cell>
                                <Flex gap={'size-100'} justifyContent='end' alignItems='center'>
                                    <ActionButton
                                        isQuiet
                                        isDisabled={areWeightsDeleted}
                                        aria-label={`Download model ${variant.id}`}
                                        onPress={() => handleDownloadModel(variant)}
                                    >
                                        <DownloadIcon />
                                    </ActionButton>
                                </Flex>
                            </Cell>
                        </Row>
                    );
                }}
            </TableBody>
        </TableView>
    );
};
