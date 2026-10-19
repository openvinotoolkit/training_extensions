// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { mapProjectLabels, UNMAPPED_LABEL_VALUE } from './util';

describe('util - label mapping', () => {
    it('returns an empty mapping when datasetLabels is empty', () => {
        const formData = new FormData();
        const mapping = mapProjectLabels([], formData);

        expect(mapping).toEqual({});
    });

    it('maps labels when corresponding target labels are provided', () => {
        const datasetLabels = ['cat', 'dog', 'bird'];
        const formData = new FormData();
        formData.set('targetLabel-0', 'feline');
        formData.set('targetLabel-1', 'canine');
        formData.set('targetLabel-2', 'avian');

        const mapping = mapProjectLabels(datasetLabels, formData);

        expect(mapping).toEqual({
            cat: 'feline',
            dog: 'canine',
            bird: 'avian',
        });
    });

    it('skips labels when the target label is missing', () => {
        const datasetLabels = ['cat', 'dog'];
        const formData = new FormData();
        formData.set('targetLabel-0', 'feline');
        // targetLabel-1 is intentionally not set

        const mapping = mapProjectLabels(datasetLabels, formData);

        expect(mapping).toEqual({
            cat: 'feline',
            dog: null,
        });
    });

    it('skips labels when the target label is an empty string', () => {
        const datasetLabels = ['cat', 'dog'];
        const formData = new FormData();
        formData.set('targetLabel-0', '');
        formData.set('targetLabel-1', 'canine');

        const mapping = mapProjectLabels(datasetLabels, formData);

        expect(mapping).toEqual({
            cat: null,
            dog: 'canine',
        });
    });

    it('skips labels when the target label is a File (non-string)', () => {
        const datasetLabels = ['image'];
        const formData = new FormData();
        const file = new File(['content'], 'image.txt', { type: 'text/plain' });
        formData.set('targetLabel-0', file);

        const mapping = mapProjectLabels(datasetLabels, formData);

        expect(mapping).toEqual({ image: null });
    });

    it('treats the unmapped sentinel value as no selection', () => {
        const datasetLabels = ['cat', 'dog'];
        const formData = new FormData();
        formData.set('targetLabel-0', UNMAPPED_LABEL_VALUE);
        formData.set('targetLabel-1', 'canine');

        const mapping = mapProjectLabels(datasetLabels, formData);

        expect(mapping).toEqual({
            cat: null,
            dog: 'canine',
        });
    });

    it('preserves duplicate source labels by keeping the last mapping', () => {
        const datasetLabels = ['cat', 'cat'];
        const formData = new FormData();
        formData.set('targetLabel-0', 'feline');
        formData.set('targetLabel-1', 'kitty');

        const mapping = mapProjectLabels(datasetLabels, formData);

        expect(mapping).toEqual({ cat: 'kitty' });
    });

    it('ignores form entries that do not correspond to dataset label indices', () => {
        const datasetLabels = ['cat'];
        const formData = new FormData();
        formData.set('targetLabel-0', 'feline');
        formData.set('targetLabel-1', 'canine');
        formData.set('unrelatedField', 'value');

        const mapping = mapProjectLabels(datasetLabels, formData);

        expect(mapping).toEqual({ cat: 'feline' });
    });

    it('handles sparse mappings correctly', () => {
        const datasetLabels = ['cat', 'dog', 'bird', 'fish'];
        const formData = new FormData();
        formData.set('targetLabel-0', 'feline');
        // skip index 1
        formData.set('targetLabel-2', 'avian');
        // skip index 3

        const mapping = mapProjectLabels(datasetLabels, formData);

        expect(mapping).toEqual({
            cat: 'feline',
            bird: 'avian',
            dog: null,
            fish: null,
        });
    });
});
