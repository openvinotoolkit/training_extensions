// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { waitFor } from '@testing-library/react';
import { getMockedModel, getMockedModelArchitecture } from 'mocks/mock-model';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';
import { renderHook } from 'test-utils/render';

import { http } from '../../../../api/utils';
import { paths } from '../../../../constants/paths';
import { server } from '../../../../msw-node-setup';
import { useResolvedModelArchitectures } from './use-resolved-model-architectures.hook';

const PROJECT_ID = '7b073838-99d3-42ff-9018-4e901eb047fc';
const STATIC_ARCHITECTURE_ID = 'object-detection-yolox-x';
const TIMM_ARCHITECTURE_ID = 'image-classification-timm-resnet50.a1_in1k';

const renderResolvedArchitectures = (models: Parameters<typeof useResolvedModelArchitectures>[0]) =>
    renderHook(() => useResolvedModelArchitectures(models), {
        route: paths.project.models({ projectId: PROJECT_ID }),
        path: paths.project.models.pattern,
    });

describe('useResolvedModelArchitectures', () => {
    beforeEach(() => {
        server.use(
            http.get('/api/projects/{project_id}', () => HttpResponse.json(getMockedProject({ id: PROJECT_ID }))),
            http.get('/api/model_architectures', () =>
                HttpResponse.json({
                    model_architectures: [getMockedModelArchitecture({ id: STATIC_ARCHITECTURE_ID, name: 'YOLOX-X' })],
                    top_picks: null,
                })
            )
        );
    });

    it('returns the static architectures unchanged when every model is covered', async () => {
        const { result } = renderResolvedArchitectures([getMockedModel({ architecture: STATIC_ARCHITECTURE_ID })]);

        await waitFor(() => {
            expect(result.current).toHaveLength(1);
        });
        expect(result.current[0].id).toBe(STATIC_ARCHITECTURE_ID);
    });

    it('resolves a timm architecture ID missing from the static list, license included', async () => {
        server.use(
            http.get('/api/model_architectures/{architecture_id}', ({ params }) =>
                HttpResponse.json(
                    getMockedModelArchitecture({
                        id: String(params['architecture_id']),
                        name: 'resnet50.a1_in1k',
                        license: { name: 'apache-2.0', url: 'https://example.com/apache-2.0' },
                    })
                )
            )
        );

        const { result } = renderResolvedArchitectures([
            getMockedModel({ architecture: STATIC_ARCHITECTURE_ID }),
            getMockedModel({ architecture: TIMM_ARCHITECTURE_ID }),
        ]);

        await waitFor(() => {
            expect(result.current).toHaveLength(2);
        });

        const timmArchitecture = result.current.find(({ id }) => id === TIMM_ARCHITECTURE_ID);
        expect(timmArchitecture?.name).toBe('resnet50.a1_in1k');
        expect(timmArchitecture?.license.name).toBe('apache-2.0');
    });

    it('leaves unknown architecture IDs unresolved when the by-ID lookup fails', async () => {
        server.use(
            http.get('/api/model_architectures/{architecture_id}', () => new HttpResponse(null, { status: 404 }))
        );

        const { result } = renderResolvedArchitectures([
            getMockedModel({ architecture: STATIC_ARCHITECTURE_ID }),
            getMockedModel({ architecture: TIMM_ARCHITECTURE_ID }),
        ]);

        // Only the static architecture is present; the row falls back to the raw ID as before
        await waitFor(() => {
            expect(result.current).toHaveLength(1);
        });
        expect(result.current[0].id).toBe(STATIC_ARCHITECTURE_ID);
    });
});
