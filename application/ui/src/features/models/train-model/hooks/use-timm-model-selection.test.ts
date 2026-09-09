// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { act, waitFor } from '@testing-library/react';
import { getMockedModelArchitecture } from 'mocks/mock-model';
import { HttpResponse } from 'msw';
import { renderHook } from 'test-utils/render';

import { http } from '../../../../api/utils';
import { server } from '../../../../msw-node-setup';
import { useTimmModelSelection } from './use-timm-model-selection';

const TIMM_CATALOG: Record<string, Record<string, string[]>> = {
    resnet: {
        resnet50: ['a1_in1k', 'gluon_in1k'],
        resnet101: ['a1h_in1k'],
    },
    efficientnet: {
        efficientnet_b0: ['ra_in1k'],
    },
};

const registerCatalogHandlers = () => {
    server.use(
        http.get('/api/model_architectures/timm/families', () => HttpResponse.json(Object.keys(TIMM_CATALOG))),
        http.get('/api/model_architectures/timm/families/{family}/variants', ({ params }) =>
            HttpResponse.json(Object.keys(TIMM_CATALOG[params.family] ?? {}))
        ),
        http.get('/api/model_architectures/timm/families/{family}/variants/{variant}/pretrained-tags', ({ params }) =>
            HttpResponse.json(TIMM_CATALOG[params.family]?.[params.variant] ?? [])
        ),
        http.get('/api/model_architectures/timm/manifest', ({ request }) => {
            const query = new URL(request.url).searchParams;

            return HttpResponse.json(
                getMockedModelArchitecture({
                    id: `image-classification-timm-${query.get('variant')}.${query.get('pretrained_tag')}`,
                })
            );
        })
    );
};

describe('useTimmModelSelection', () => {
    beforeEach(registerCatalogHandlers);

    it('does not fetch the catalog while disabled', async () => {
        const { result } = renderHook(() => useTimmModelSelection(false));

        // Give the async path a chance to fire — it should not.
        await new Promise((resolve) => setTimeout(resolve, 50));
        expect(result.current.timmFamilies).toEqual([]);

        expect(result.current.timmModelArchitecture).toBeUndefined();
    });

    it('defaults the variant and the pretrained tag once a family is selected', async () => {
        const { result } = renderHook(() => useTimmModelSelection(true));

        await waitFor(() => {
            expect(result.current.timmFamilies).toEqual(['resnet', 'efficientnet']);
        });

        act(() => {
            result.current.onSelectTimmFamily('resnet');
        });

        await waitFor(() => {
            expect(result.current.selectedTimmVariant).toBe('resnet50');
        });

        await waitFor(() => {
            expect(result.current.selectedTimmPretrainedTag).toBe('a1_in1k');
        });

        await waitFor(() => {
            expect(result.current.timmModelArchitecture?.id).toBe('image-classification-timm-resnet50.a1_in1k');
        });
    });

    it('resets the pretrained tag when the variant changes', async () => {
        const { result } = renderHook(() => useTimmModelSelection(true));

        act(() => {
            result.current.onSelectTimmFamily('resnet');
        });

        await waitFor(() => {
            expect(result.current.selectedTimmPretrainedTag).toBe('a1_in1k');
        });

        act(() => {
            result.current.onSelectTimmVariant('resnet101');
        });

        await waitFor(() => {
            expect(result.current.selectedTimmPretrainedTag).toBe('a1h_in1k');
        });

        await waitFor(() => {
            expect(result.current.timmModelArchitecture?.id).toBe('image-classification-timm-resnet101.a1h_in1k');
        });
    });

    it('resets the variant and the pretrained tag when the family changes', async () => {
        const { result } = renderHook(() => useTimmModelSelection(true));

        act(() => {
            result.current.onSelectTimmFamily('resnet');
        });

        await waitFor(() => {
            expect(result.current.selectedTimmVariant).toBe('resnet50');
        });

        act(() => {
            result.current.onSelectTimmFamily('efficientnet');
        });

        await waitFor(() => {
            expect(result.current.selectedTimmVariant).toBe('efficientnet_b0');
        });

        await waitFor(() => {
            expect(result.current.selectedTimmPretrainedTag).toBe('ra_in1k');
        });
    });
});
