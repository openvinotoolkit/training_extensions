// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, screen } from '@testing-library/react';
import { getMockedAnnotation } from 'mocks/mock-annotation';
import { render } from 'test-utils/render';

import type { Annotation } from '../../../shared/types';
import { useSelectNextAnnotation, type UseSelectNextAnnotationProps } from './use-select-next-annotation.hook';

const App = (props: UseSelectNextAnnotationProps) => {
    useSelectNextAnnotation(props);

    return <div data-testid='target' />;
};

const pressTab = () => {
    fireEvent.keyDown(screen.getByTestId('target'), { key: 'Tab', code: 'Tab' });
};

describe('useSelectNextAnnotation', () => {
    const annotations = [
        getMockedAnnotation({ id: 'annotation-1' }),
        getMockedAnnotation({ id: 'annotation-2' }),
        getMockedAnnotation({ id: 'annotation-3' }),
    ];

    it.each([
        { name: 'selects the next annotation', annotations, selectedId: 'annotation-1', expectedId: 'annotation-2' },
        {
            name: 'wraps around to the first annotation when the last one is selected',
            annotations,
            selectedId: 'annotation-3',
            expectedId: 'annotation-1',
        },
        {
            name: 'keeps the selection when there is a single annotation',
            annotations: [annotations[0]],
            selectedId: 'annotation-1',
            expectedId: 'annotation-1',
        },
    ])('$name on Tab', ({ annotations: items, selectedId, expectedId }) => {
        const updateSelectedAnnotationsIds = vi.fn();
        render(
            <App
                annotations={items}
                selectedAnnotationsIds={new Set([selectedId])}
                updateSelectedAnnotationsIds={updateSelectedAnnotationsIds}
            />
        );

        pressTab();

        expect(updateSelectedAnnotationsIds).toHaveBeenCalledExactlyOnceWith(new Set([expectedId]));
    });

    it.each<{ name: string; annotations: Annotation[]; selectedIds: string[] }>([
        { name: 'no annotation is selected', annotations, selectedIds: [] },
        { name: 'multiple annotations are selected', annotations, selectedIds: ['annotation-1', 'annotation-2'] },
        {
            name: 'the selected annotation is not part of the annotations',
            annotations,
            selectedIds: ['annotation-from-another-media-item'],
        },
        { name: 'there are no annotations', annotations: [], selectedIds: ['annotation-1'] },
    ])('does not change the selection when $name', ({ annotations: items, selectedIds }) => {
        const updateSelectedAnnotationsIds = vi.fn();
        render(
            <App
                annotations={items}
                selectedAnnotationsIds={new Set(selectedIds)}
                updateSelectedAnnotationsIds={updateSelectedAnnotationsIds}
            />
        );

        pressTab();

        expect(updateSelectedAnnotationsIds).not.toHaveBeenCalled();
    });
});
