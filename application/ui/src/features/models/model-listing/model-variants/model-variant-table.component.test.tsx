// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { getMockedModel } from 'mocks/mock-model';
import { getMockedVariant } from 'mocks/mock-model-variant';
import { render } from 'test-utils/render';

import { downloadFile } from '../../../../platform/download-file';
import { ModelVariantTable } from './model-variant-table.component';

vi.mock('../../../../platform/download-file', () => ({
    downloadFile: vi.fn(),
}));

vi.mock('hooks/use-project-identifier.hook', () => ({
    useProjectIdentifier: () => 'project-123',
}));

describe('ModelVariantTable', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('shows primary testing metric value for a variant', () => {
        const model = getMockedModel({
            variants: [
                getMockedVariant({
                    id: 'ov-1',
                    format: 'openvino',
                    precision: 'fp16',
                    evaluations: [
                        {
                            dataset_revision_id: 'dataset-1',
                            subset: 'testing',
                            metrics: [{ name: 'Accuracy', value: 0.923, primary: true }],
                        },
                    ],
                }),
            ],
        });

        render(<ModelVariantTable model={model} format='openvino' />);

        expect(screen.getByRole('columnheader', { name: 'Accuracy' })).toBeInTheDocument();
        expect(screen.getByTestId('model-variant-value-accuracy-fp16')).toHaveTextContent('92%');
    });

    it('falls back to fp32 pytorch primary metric when evaluations are empty', () => {
        const model = getMockedModel({
            variants: [
                getMockedVariant({
                    id: 'ov-1',
                    format: 'openvino',
                    precision: 'fp16',
                    evaluations: [],
                }),
                getMockedVariant({
                    id: 'pt-1',
                    format: 'pytorch',
                    precision: 'fp32',
                    evaluations: [
                        {
                            dataset_revision_id: 'dataset-1',
                            subset: 'testing',
                            metrics: [{ name: 'mAP', value: 0.87, primary: true }],
                        },
                    ],
                }),
            ],
        });

        render(<ModelVariantTable model={model} format='openvino' />);

        expect(screen.getByRole('columnheader', { name: 'mAP' })).toBeInTheDocument();
        expect(screen.getByTestId('model-variant-value-accuracy-fp16')).toHaveTextContent('87%');
    });

    it('shows size and performance deltas for the quantized variant', () => {
        const model = getMockedModel({
            variants: [
                getMockedVariant({
                    id: 'ov-fp16',
                    format: 'openvino',
                    precision: 'fp16',
                    weights_size: 1000,
                    evaluations: [
                        {
                            dataset_revision_id: 'dataset-1',
                            subset: 'testing',
                            metrics: [{ name: 'Accuracy', value: 0.92, primary: true }],
                        },
                    ],
                }),
                getMockedVariant({
                    id: 'ov-int8',
                    format: 'openvino',
                    precision: 'int8',
                    weights_size: 500,
                    evaluations: [
                        {
                            dataset_revision_id: 'dataset-1',
                            subset: 'testing',
                            metrics: [{ name: 'Accuracy', value: 0.89, primary: true }],
                        },
                    ],
                }),
            ],
        });

        render(<ModelVariantTable model={model} format='openvino' />);

        expect(screen.getByText('-50%')).toHaveStyle({ color: 'var(--moss-tint-1)' });
        expect(screen.getByText('-3%')).toHaveStyle({ color: 'var(--coral-shade-1)' });
    });

    it('does not fallback value when variant has evaluations but no primary metric', () => {
        const model = getMockedModel({
            variants: [
                getMockedVariant({
                    id: 'ov-1',
                    format: 'openvino',
                    precision: 'fp16',
                    evaluations: [
                        {
                            dataset_revision_id: 'dataset-1',
                            subset: 'testing',
                            metrics: [{ name: 'Accuracy', value: 0.91, primary: false }],
                        },
                    ],
                }),
                getMockedVariant({
                    id: 'pt-1',
                    format: 'pytorch',
                    precision: 'fp32',
                    evaluations: [
                        {
                            dataset_revision_id: 'dataset-1',
                            subset: 'testing',
                            metrics: [{ name: 'mAP', value: 0.87, primary: true }],
                        },
                    ],
                }),
            ],
        });

        render(<ModelVariantTable model={model} format='openvino' />);

        expect(screen.getByRole('columnheader', { name: 'mAP' })).toBeInTheDocument();
        expect(screen.getByTestId('model-variant-value-accuracy-fp16')).toHaveTextContent('-');
        expect(screen.queryByText('87%')).not.toBeInTheDocument();
    });

    it('shows Accuracy and dash when no primary metric is available anywhere', () => {
        const model = getMockedModel({
            variants: [
                getMockedVariant({
                    id: 'ov-1',
                    format: 'openvino',
                    precision: 'fp16',
                    evaluations: [],
                }),
            ],
        });

        render(<ModelVariantTable model={model} format='openvino' />);

        expect(screen.getByRole('columnheader', { name: 'Accuracy' })).toBeInTheDocument();
        expect(screen.getByTestId('model-variant-value-accuracy-fp16')).toHaveTextContent('-');
    });

    it('triggers download for selected variant', async () => {
        const model = getMockedModel({
            variants: [
                getMockedVariant({
                    id: 'ov-1',
                    format: 'openvino',
                    precision: 'fp16',
                }),
            ],
        });

        render(<ModelVariantTable model={model} format='openvino' />);

        await userEvent.click(screen.getByRole('button', { name: 'Download model ov-1' }));

        expect(downloadFile).toHaveBeenCalledWith(
            expect.stringContaining(`/api/projects/project-123/models/${model.id}/variants/ov-1/binary`),
            `model-${String(model.id).split('-')[0]}-openvino-fp16.zip`,
            'Model download started'
        );
    });

    it('shows direct download button for non-openvino format', () => {
        const model = getMockedModel({
            variants: [
                getMockedVariant({
                    id: 'pt-1',
                    format: 'pytorch',
                    precision: 'fp32',
                }),
            ],
        });

        render(<ModelVariantTable model={model} format='pytorch' />);

        expect(screen.getByRole('button', { name: 'Download model pt-1' })).toBeInTheDocument();
    });

    it.each(['pytorch', 'onnx'] as const)('%s format renders a direct download button', (format) => {
        const model = getMockedModel({
            variants: [
                getMockedVariant({
                    id: `${format}-1`,
                    format,
                    precision: 'fp32',
                    quantization_info: null,
                }),
            ],
        });

        render(<ModelVariantTable model={model} format={format} />);

        expect(screen.getByRole('button', { name: `Download model ${format}-1` })).toBeInTheDocument();
    });

    it.each([
        ['the model weights were deleted', { modelFilesDeleted: true, variantFilesDeleted: false }],
        ['the variant weights were deleted', { modelFilesDeleted: false, variantFilesDeleted: true }],
    ])('shows a dash size and a disabled download button when %s', (_, { modelFilesDeleted, variantFilesDeleted }) => {
        const model = getMockedModel({
            files_deleted: modelFilesDeleted,
            variants: [
                getMockedVariant({
                    id: 'ov-1',
                    format: 'openvino',
                    precision: 'fp16',
                    weights_size: 0,
                    files_deleted: variantFilesDeleted,
                    evaluations: [
                        {
                            dataset_revision_id: 'dataset-1',
                            subset: 'testing',
                            metrics: [{ name: 'Accuracy', value: 0.923, primary: true }],
                        },
                    ],
                }),
            ],
        });

        render(<ModelVariantTable model={model} format='openvino' />);

        expect(screen.getByTestId('model-variant-value-size-fp16')).toHaveTextContent('-');
        expect(screen.getByRole('button', { name: 'Download model ov-1' })).toBeDisabled();

        // Precision and score remain visible for the record
        expect(screen.getByText('FP16')).toBeInTheDocument();
        expect(screen.getByTestId('model-variant-value-accuracy-fp16')).toHaveTextContent('92%');
    });

    describe('ModelVariantPrecisionRenderer', () => {
        it('renders only uppercased precision text when variant has no quantization_info', () => {
            const model = getMockedModel({
                variants: [
                    getMockedVariant({
                        format: 'openvino',
                        precision: 'fp16',
                        quantization_info: null,
                    }),
                ],
            });

            render(<ModelVariantTable model={model} format='openvino' />);

            expect(screen.getByText('FP16')).toBeInTheDocument();
            expect(screen.queryByRole('button', { name: 'Information' })).not.toBeInTheDocument();
        });

        it('renders precision text and contextual help popover with max_drop and calibration size when quantization_info is present', async () => {
            const model = getMockedModel({
                variants: [
                    getMockedVariant({
                        format: 'openvino',
                        precision: 'int8',
                        quantization_info: { max_drop: 0.02, max_calibration_subset_size: 100 },
                    }),
                ],
            });

            render(<ModelVariantTable model={model} format='openvino' />);

            expect(screen.getByText('INT8')).toBeInTheDocument();

            const infoButton = screen.getByRole('button', { name: 'Information' });
            expect(infoButton).toBeInTheDocument();

            await userEvent.click(infoButton);

            expect(screen.getByText('Quantized with NNCF PTQ')).toBeInTheDocument();
            expect(screen.getByText('Max accuracy drop: 2%')).toBeInTheDocument();
            expect(screen.getByText('Calibration dataset size: 100')).toBeInTheDocument();
        });

        it('does not render max accuracy drop line when max_drop is null', async () => {
            const model = getMockedModel({
                variants: [
                    getMockedVariant({
                        format: 'openvino',
                        precision: 'int8',
                        quantization_info: { max_drop: null, max_calibration_subset_size: 50 },
                    }),
                ],
            });

            render(<ModelVariantTable model={model} format='openvino' />);

            await userEvent.click(screen.getByRole('button', { name: 'Information' }));

            expect(screen.queryByText(/Max accuracy drop/)).not.toBeInTheDocument();
            expect(screen.getByText('Calibration dataset size: 50')).toBeInTheDocument();
        });

        it('renders max accuracy drop as 0% when max_drop is zero', async () => {
            const model = getMockedModel({
                variants: [
                    getMockedVariant({
                        format: 'openvino',
                        precision: 'int8',
                        quantization_info: { max_drop: 0, max_calibration_subset_size: 200 },
                    }),
                ],
            });

            render(<ModelVariantTable model={model} format='openvino' />);

            await userEvent.click(screen.getByRole('button', { name: 'Information' }));

            expect(screen.getByText('Max accuracy drop: 0%')).toBeInTheDocument();
            expect(screen.getByText('Calibration dataset size: 200')).toBeInTheDocument();
        });

        it('does not render contextual help when both max_drop and calibration_subset_size are null', () => {
            const model = getMockedModel({
                variants: [
                    getMockedVariant({
                        format: 'openvino',
                        precision: 'int8',
                        quantization_info: { max_drop: null, max_calibration_subset_size: null },
                    }),
                ],
            });

            render(<ModelVariantTable model={model} format='openvino' />);

            expect(screen.getByText('INT8')).toBeInTheDocument();
            expect(screen.queryByRole('button', { name: 'Information' })).not.toBeInTheDocument();
        });
    });
});
