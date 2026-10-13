// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fromOpenApi } from '@msw/source/open-api';
import { createOpenApiHttp, OpenApiHttpHandlers } from 'openapi-msw';

import type { paths } from './openapi-spec';
import spec from './openapi-spec.json' with { type: 'json' };

const handlers = await fromOpenApi(JSON.stringify(spec).replace(/}:/g, '}//:'));

const getOpenApiHttp = (): OpenApiHttpHandlers<paths> => {
    const http = createOpenApiHttp<paths>({
        baseUrl: process.env.PUBLIC_API_BASE_URL || '',
    });

    return {
        ...http,
        post: (path, ...other) => {
            // @ts-expect-error MSW internal parsing function does not accept paths like
            // `/api/models/{model_name}:activate` or `/api/staged_datasets:from-upload`
            // to get around this we escape the colon character with `\\`. OpenAPI paths use
            // `{param}` placeholders, so every colon is a literal custom-method separator.
            // @see https://github.com/mswjs/msw/discussions/739
            return http.post(path.replaceAll(':', '\\:'), ...other);
        },
    };
};

const http = getOpenApiHttp();

export { handlers, http };
