// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { renderHook } from 'test-utils/render';
import { vi } from 'vitest';

import { useTrainModelDisabledReason } from './use-train-model-disabled-reason';

const mockUseGetDatasetItems = vi.hoisted(() => vi.fn());

vi.mock('hooks/use-get-dataset-items.hook', () => ({
    useGetDatasetItems: mockUseGetDatasetItems,
}));

const mockDatasetItems = (
    counts: {
        total: number;
        training: number;
        testing: number;
        validation: number;
        reviewedUnassigned: number;
        unassigned: number;
    },
    isPending = false
) => {
    mockUseGetDatasetItems
        .mockReturnValueOnce({ totalCount: counts.total, isPending })
        .mockReturnValueOnce({ totalCount: counts.training, isPending })
        .mockReturnValueOnce({ totalCount: counts.testing, isPending })
        .mockReturnValueOnce({ totalCount: counts.validation, isPending })
        .mockReturnValueOnce({ totalCount: counts.reviewedUnassigned, isPending })
        .mockReturnValueOnce({ totalCount: counts.unassigned, isPending });
};

describe('useTrainModelDisabledReason', () => {
    beforeEach(() => {
        mockUseGetDatasetItems.mockReset();
    });

    it('returns undefined reason when queries are pending', () => {
        mockDatasetItems(
            { total: 0, training: 0, testing: 0, validation: 0, reviewedUnassigned: 0, unassigned: 0 },
            true
        );

        const { result } = renderHook(() => useTrainModelDisabledReason());

        expect(result.current.reason).toBeUndefined();
    });

    it('returns reason when total annotated items is less than 3', () => {
        mockDatasetItems({ total: 2, training: 1, testing: 1, validation: 0, reviewedUnassigned: 0, unassigned: 0 });

        const { result } = renderHook(() => useTrainModelDisabledReason());

        expect(result.current.reason).toBe(
            'In order to train a model, you need to annotate at least 3 items in your dataset, although we ' +
                'recommend annotating several more for better results.'
        );
    });

    it('returns undefined reason when exactly 3 annotations and all subsets non-empty', () => {
        mockDatasetItems({ total: 3, training: 1, testing: 1, validation: 1, reviewedUnassigned: 0, unassigned: 0 });

        const { result } = renderHook(() => useTrainModelDisabledReason());

        expect(result.current.reason).toBeUndefined();
    });

    describe('empty subsets', () => {
        it('mentions training subset when training is empty', () => {
            mockDatasetItems({
                total: 5,
                training: 0,
                testing: 1,
                validation: 1,
                reviewedUnassigned: 0,
                unassigned: 0,
            });

            const { result } = renderHook(() => useTrainModelDisabledReason());

            expect(result.current.reason).toBe(
                'In order to train a model, each subset (training, validation, testing) needs at least one item. ' +
                    'This condition is currently not satisfiable, because the training subset is empty and ' +
                    'there are no unassigned items available to redistribute.'
            );
        });

        it('includes both reviewed and unannotated unassigned detail when both are present', () => {
            mockDatasetItems({
                total: 4,
                training: 0,
                testing: 0,
                validation: 0,
                reviewedUnassigned: 1,
                unassigned: 3,
            });

            const { result } = renderHook(() => useTrainModelDisabledReason());

            expect(result.current.reason).toBe(
                'In order to train a model, each subset (training, validation, testing) needs at least one item. ' +
                    'This condition is currently not satisfiable, because the training, validation, and testing subsets are empty and ' +
                    'there is 1 reviewed item ready to assign and ' +
                    '2 items that still need annotation before they can be assigned.'
            );
        });

        it('mentions both reviewed and unannotated items with singular counts on each side', () => {
            mockDatasetItems({
                total: 4,
                training: 0,
                testing: 1,
                validation: 0,
                reviewedUnassigned: 1,
                unassigned: 2,
            });

            const { result } = renderHook(() => useTrainModelDisabledReason());

            expect(result.current.reason).toBe(
                'In order to train a model, each subset (training, validation, testing) needs at least one item. ' +
                    'This condition is currently not satisfiable, because the training and validation subsets are empty and ' +
                    'there is 1 reviewed item ready to assign and 1 item that still needs annotation before it can be assigned.'
            );
        });

        it('mentions only unannotated items when no reviewed unassigned items exist', () => {
            mockDatasetItems({
                total: 4,
                training: 0,
                testing: 1,
                validation: 0,
                reviewedUnassigned: 0,
                unassigned: 3,
            });

            const { result } = renderHook(() => useTrainModelDisabledReason());

            expect(result.current.reason).toBe(
                'In order to train a model, each subset (training, validation, testing) needs at least one item. ' +
                    'This condition is currently not satisfiable, because the training and validation subsets are empty and ' +
                    'there are 3 items that still need annotation before they can be assigned.'
            );
        });

        it('uses singular form for unannotated items when count is 1', () => {
            mockDatasetItems({
                total: 4,
                training: 0,
                testing: 1,
                validation: 0,
                reviewedUnassigned: 0,
                unassigned: 1,
            });

            const { result } = renderHook(() => useTrainModelDisabledReason());

            expect(result.current.reason).toBe(
                'In order to train a model, each subset (training, validation, testing) needs at least one item. ' +
                    'This condition is currently not satisfiable, because the training and validation subsets are empty and ' +
                    'there is 1 item that still needs annotation before it can be assigned.'
            );
        });

        it('returns undefined reason when empty subsets count does not exceed reviewed unassigned items', () => {
            mockDatasetItems({
                total: 4,
                training: 0,
                testing: 1,
                validation: 1,
                reviewedUnassigned: 1,
                unassigned: 1,
            });

            const { result } = renderHook(() => useTrainModelDisabledReason());

            expect(result.current.reason).toBeUndefined();
        });

        it('returns undefined reason when two subsets are empty but enough reviewed unassigned items to cover both', () => {
            mockDatasetItems({
                total: 5,
                training: 0,
                testing: 1,
                validation: 0,
                reviewedUnassigned: 2,
                unassigned: 2,
            });

            const { result } = renderHook(() => useTrainModelDisabledReason());

            expect(result.current.reason).toBeUndefined();
        });

        it('uses plural form when two subsets are empty', () => {
            mockDatasetItems({
                total: 5,
                training: 0,
                testing: 1,
                validation: 0,
                reviewedUnassigned: 0,
                unassigned: 0,
            });

            const { result } = renderHook(() => useTrainModelDisabledReason());

            expect(result.current.reason).toBe(
                'In order to train a model, each subset (training, validation, testing) needs at least one item. ' +
                    'This condition is currently not satisfiable, because the training and validation subsets are empty and ' +
                    'there are no unassigned items available to redistribute.'
            );
        });

        it('uses comma list format when all three subsets are empty', () => {
            mockDatasetItems({
                total: 5,
                training: 0,
                testing: 0,
                validation: 0,
                reviewedUnassigned: 0,
                unassigned: 0,
            });

            const { result } = renderHook(() => useTrainModelDisabledReason());

            expect(result.current.reason).toBe(
                'In order to train a model, each subset (training, validation, testing) needs at least one item. ' +
                    'This condition is currently not satisfiable, because the training, validation, and testing subsets are empty and ' +
                    'there are no unassigned items available to redistribute.'
            );
        });
    });
});
