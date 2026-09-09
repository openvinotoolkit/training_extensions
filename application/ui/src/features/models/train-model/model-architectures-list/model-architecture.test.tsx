// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { ModelArchitectureWithPerformanceCategory } from '@/api/types';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { getMockedModelArchitecture } from 'mocks/mock-model';
import { render } from 'test-utils/render';

import { DetailedModelArchitecture, ModelArchitecture } from './model-architecture.component';
import { ModelArchitecturesListLayout } from './model-architectures-list-layout/model-architectures-list-layout.component';

type OnSelectedModelArchitectureIdChange = (modelArchitectureId: string | null) => void;

const renderModelArchitecture = ({
    selectedModelArchitectureId = null,
    onSelectedModelArchitectureIdChange = vi.fn<OnSelectedModelArchitectureIdChange>(),
    modelArchitecture = getMockedModelArchitecture(),
}: {
    selectedModelArchitectureId?: string | null;
    onSelectedModelArchitectureIdChange?: OnSelectedModelArchitectureIdChange;
    modelArchitecture?: ModelArchitectureWithPerformanceCategory;
} = {}) => {
    render(
        <ModelArchitecturesListLayout
            selectedModelArchitectureId={selectedModelArchitectureId}
            onSelectedModelArchitectureIdChange={onSelectedModelArchitectureIdChange}
            ariaLabel={'Model architectures'}
        >
            <ModelArchitecture
                modelArchitecture={modelArchitecture}
                selectedModelArchitectureId={selectedModelArchitectureId}
                onSelectedModelArchitectureIdChange={onSelectedModelArchitectureIdChange}
            />
        </ModelArchitecturesListLayout>
    );

    return { modelArchitecture };
};

const renderDetailedModelArchitecture = ({
    selectedModelArchitectureId = null,
    onSelectedModelArchitectureIdChange = vi.fn<OnSelectedModelArchitectureIdChange>(),
    modelArchitecture = getMockedModelArchitecture(),
}: {
    selectedModelArchitectureId?: string | null;
    onSelectedModelArchitectureIdChange?: OnSelectedModelArchitectureIdChange;
    modelArchitecture?: ModelArchitectureWithPerformanceCategory;
} = {}) => {
    render(
        <ModelArchitecturesListLayout
            selectedModelArchitectureId={selectedModelArchitectureId}
            onSelectedModelArchitectureIdChange={onSelectedModelArchitectureIdChange}
            ariaLabel={'Model architectures'}
        >
            <DetailedModelArchitecture
                modelArchitecture={modelArchitecture}
                selectedModelArchitectureId={selectedModelArchitectureId}
                onSelectedModelArchitectureIdChange={onSelectedModelArchitectureIdChange}
            />
        </ModelArchitecturesListLayout>
    );

    return { modelArchitecture };
};

describe('ModelArchitecture', () => {
    it('renders the model architecture name and number of parameters', () => {
        const modelArchitecture = getMockedModelArchitecture({ name: 'Deim-DFine-L' });
        renderModelArchitecture({ modelArchitecture });

        expect(screen.getByText(modelArchitecture.name)).toBeVisible();
        expect(
            screen.getByText(`Number of parameters: ${modelArchitecture.stats?.trainable_parameters} million`)
        ).toBeVisible();
    });

    it('does not render a performance category badge when performanceCategory is undefined', () => {
        renderModelArchitecture({ modelArchitecture: getMockedModelArchitecture({ performanceCategory: undefined }) });

        expect(screen.queryByText(/speed|balance|accuracy/i)).not.toBeInTheDocument();
    });

    it('renders the performance category badge when performanceCategory is defined', () => {
        renderModelArchitecture({ modelArchitecture: getMockedModelArchitecture({ performanceCategory: 'speed' }) });

        expect(screen.getByText('Speed')).toBeVisible();
    });

    it('calls onSelectedModelArchitectureIdChange with the architecture id when clicked', async () => {
        const user = userEvent.setup();
        const onSelectedModelArchitectureIdChange = vi.fn();

        renderModelArchitecture({ onSelectedModelArchitectureIdChange });

        await user.click(screen.getByText('Deim-DFine-L'));

        expect(onSelectedModelArchitectureIdChange).toHaveBeenCalledWith('Object_Detection_Deim_DFine_L');
    });

    it('renders the radio button as selected when selectedModelArchitectureId matches', () => {
        renderModelArchitecture({
            selectedModelArchitectureId: 'Object_Detection_Deim_DFine_L',
            modelArchitecture: getMockedModelArchitecture({ id: 'Object_Detection_Deim_DFine_L' }),
        });

        expect(screen.getByRole('radio', { name: 'Deim-DFine-L' })).toBeChecked();
    });

    it('renders the radio button as not selected when selectedModelArchitectureId does not match', () => {
        renderModelArchitecture({
            selectedModelArchitectureId: 'some-other-id',
            modelArchitecture: getMockedModelArchitecture({ id: 'Object_Detection_Deim_DFine_L' }),
        });

        expect(screen.getByRole('radio', { name: 'Deim-DFine-L' })).not.toBeChecked();
    });

    it('renders the radio button as not selected when selectedModelArchitectureId is null', () => {
        renderModelArchitecture({ selectedModelArchitectureId: null });

        expect(screen.getByRole('radio', { name: 'Deim-DFine-L' })).not.toBeChecked();
    });

    describe('benchmark stats (DetailedParameters)', () => {
        const detectionArchitecture = getMockedModelArchitecture({
            task: 'detection',
            stats: {
                gigaflops: 91,
                trainable_parameters: 31,
                benchmark_metrics: {
                    imagenet_top1_accuracy: null,
                    imagenet_top5_accuracy: null,
                    coco_map_50_95: 55.3,
                    coco_map_50: null,
                },
            },
            performanceCategory: 'speed',
        });

        it('does not show gigaflops or accuracy when using Parameters (default)', () => {
            renderModelArchitecture({ modelArchitecture: detectionArchitecture });

            expect(screen.queryByText(/gigaflops/i)).not.toBeInTheDocument();
            expect(screen.queryByText(/mAP/i)).not.toBeInTheDocument();
        });

        it('shows gigaflops and mAP when using DetailedParameters', () => {
            renderDetailedModelArchitecture({ modelArchitecture: detectionArchitecture });

            expect(screen.getByText(`Gigaflops: ${detectionArchitecture.stats?.gigaflops}`)).toBeVisible();
            expect(screen.getByText('mAP on COCO: 55.3%')).toBeVisible();
        });

        it('hides the performance category badge when using DetailedParameters', () => {
            renderDetailedModelArchitecture({ modelArchitecture: detectionArchitecture });

            expect(screen.queryByText('Speed')).not.toBeInTheDocument();
        });

        it('still shows the performance category badge when using Parameters only', () => {
            renderModelArchitecture({ modelArchitecture: detectionArchitecture });

            expect(screen.getByText('Speed')).toBeVisible();
        });

        it('shows Top-1 Acc for classification tasks when using DetailedParameters', () => {
            const classificationArchitecture = getMockedModelArchitecture({
                task: 'classification',
                stats: {
                    gigaflops: 2,
                    trainable_parameters: 5,
                    benchmark_metrics: {
                        imagenet_top1_accuracy: 76.2,
                        imagenet_top5_accuracy: 95.3,
                        coco_map_50_95: null,
                        coco_map_50: null,
                    },
                },
            });

            renderDetailedModelArchitecture({ modelArchitecture: classificationArchitecture });

            expect(screen.getByText('Top-1 Acc on ImageNet: 76.2%')).toBeVisible();
        });
    });
});
