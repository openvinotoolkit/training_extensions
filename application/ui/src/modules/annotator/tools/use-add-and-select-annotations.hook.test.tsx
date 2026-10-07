// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { type ReactNode } from 'react';

import { act } from '@testing-library/react';
import { getMockedShape } from 'mocks/mock-annotation';
import { getMockedLabel } from 'mocks/mock-labels';

import { renderHook } from '../../../test-utils/render';
import { AnnotationDocumentProvider, useAnnotations } from '../annotation-document-provider.component';
import { SelectAnnotationProvider, useSelectedAnnotations } from '../select-annotation-provider.component';
import { useAddAndSelectAnnotations } from './use-add-and-select-annotations.hook';

const label = getMockedLabel({ id: 'label-1' });

const wrapper = ({ children }: { children: ReactNode }) => (
    <AnnotationDocumentProvider mode='annotation' initialAnnotationsDTO={[]} initialPredictionsDTO={[]}>
        <SelectAnnotationProvider>{children}</SelectAnnotationProvider>
    </AnnotationDocumentProvider>
);

describe('useAddAndSelectAnnotations', () => {
    it('adds and selects the new annotations', () => {
        const { result } = renderHook(
            () => ({ ...useAddAndSelectAnnotations(), ...useAnnotations(), ...useSelectedAnnotations() }),
            { wrapper }
        );

        let newIds: string[] = [];
        act(() => {
            newIds = result.current.addAndSelectAnnotations([getMockedShape({ type: 'rectangle' })], [label]);
        });

        expect(result.current.annotations.map(({ id }) => id)).toEqual(newIds);
        expect(result.current.selectedAnnotations).toEqual(new Set(newIds));
    });
});
