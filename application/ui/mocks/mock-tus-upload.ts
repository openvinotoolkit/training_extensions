// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { http, HttpResponse } from 'msw';

import { getMockedMediaImage } from './mock-media';
import { getMockedStagedDataset } from './mock-staged-dataset';

type MockUpload = { length: number; offset: number; isConsumed: boolean };

const uploads = new Map<string, MockUpload>();
const protocolHeaders = {
    'Tus-Resumable': '1.0.0',
    // Component/E2E tests run the UI and API on different origins, so the TUS response headers
    // (read via the XHR/fetch Headers API, not the body) must be explicitly exposed, mirroring the
    // real backend's `CORSMiddleware` config (see `app/main.py`).
    'Access-Control-Expose-Headers':
        'Location, Tus-Resumable, Tus-Version, Tus-Extension, Tus-Max-Size, Tus-Checksum-Algorithm, ' +
        'Upload-Offset, Upload-Length, Upload-Defer-Length, Upload-Expires',
};
const capabilityHeaders = {
    ...protocolHeaders,
    'Tus-Version': '1.0.0',
    'Tus-Extension': 'creation,creation-with-upload,creation-defer-length,termination,expiration,checksum',
    'Tus-Max-Size': String(50 * 1024 ** 3),
    'Tus-Checksum-Algorithm': 'sha1,sha256,md5',
};

// Mirrors the backend's `consume` semantics: only fully transferred uploads can be claimed, once.
const consume = async (request: Request): Promise<HttpResponse<null> | undefined> => {
    const { upload_id: uploadId } = (await request.clone().json()) as { upload_id: string };
    const upload = uploads.get(uploadId);

    if (upload === undefined) {
        return new HttpResponse(null, { status: 404 });
    }
    if (upload.isConsumed) {
        return new HttpResponse(null, { status: 410 });
    }
    if (upload.offset < upload.length) {
        return new HttpResponse(null, { status: 409 });
    }

    upload.isConsumed = true;
    return undefined;
};

export const tusUploadHandlers = [
    http.options('*/api/uploads', () => new HttpResponse(null, { status: 204, headers: capabilityHeaders })),
    http.options('*/api/uploads/:uploadId', () => new HttpResponse(null, { status: 204, headers: capabilityHeaders })),
    http.post('*/api/uploads', ({ request }) => {
        const length = Number(request.headers.get('Upload-Length'));
        const uploadId = crypto.randomUUID();
        uploads.set(uploadId, { length, offset: 0, isConsumed: false });

        return new HttpResponse(null, {
            status: 201,
            headers: {
                ...protocolHeaders,
                Location: `/api/uploads/${uploadId}`,
                'Upload-Offset': '0',
                'Upload-Expires': new Date(Date.now() + 24 * 60 * 60 * 1000).toUTCString(),
            },
        });
    }),
    http.head('*/api/uploads/:uploadId', ({ params }) => {
        const upload = uploads.get(String(params.uploadId));
        if (upload === undefined) {
            return new HttpResponse(null, { status: 404, headers: protocolHeaders });
        }
        if (upload.isConsumed) {
            return new HttpResponse(null, { status: 410, headers: protocolHeaders });
        }

        return new HttpResponse(null, {
            status: 204,
            headers: {
                ...protocolHeaders,
                'Cache-Control': 'no-store',
                'Upload-Length': String(upload.length),
                'Upload-Offset': String(upload.offset),
            },
        });
    }),
    http.patch('*/api/uploads/:uploadId', async ({ params, request }) => {
        const upload = uploads.get(String(params.uploadId));
        if (upload === undefined) {
            return new HttpResponse(null, { status: 404, headers: protocolHeaders });
        }
        if (Number(request.headers.get('Upload-Offset')) !== upload.offset) {
            return new HttpResponse(null, { status: 409, headers: protocolHeaders });
        }

        upload.offset += (await request.arrayBuffer()).byteLength;

        return new HttpResponse(null, {
            status: 204,
            headers: { ...protocolHeaders, 'Upload-Offset': String(upload.offset) },
        });
    }),
    http.delete('*/api/uploads/:uploadId', ({ params }) => {
        const isDeleted = uploads.delete(String(params.uploadId));

        return new HttpResponse(null, { status: isDeleted ? 204 : 404, headers: protocolHeaders });
    }),
    http.post('*/api/projects/:projectId/dataset/media\\:from-upload', async ({ request }) => {
        return (
            (await consume(request)) ??
            HttpResponse.json(getMockedMediaImage({ id: crypto.randomUUID() }), { status: 201 })
        );
    }),
    http.post('*/api/staged_datasets\\:from-upload', async ({ request }) => {
        return (
            (await consume(request)) ??
            HttpResponse.json(getMockedStagedDataset({ id: crypto.randomUUID() }), { status: 201 })
        );
    }),
    http.post('*/api/sources/media\\:from-upload', async ({ request }) => {
        return (
            (await consume(request)) ??
            HttpResponse.json({ video_path: `/data/source_media/${crypto.randomUUID()}/sample.mp4` }, { status: 201 })
        );
    }),
];
