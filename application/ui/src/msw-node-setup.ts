// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { tusUploadHandlers } from 'mocks/mock-tus-upload';
import { setupServer } from 'msw/node';

import { handlers } from './api/utils';

// Initialize msw's mock server with the handlers. The stateful TUS handlers go first so they take
// precedence over the example responses generated from the OpenAPI spec.
export const server = setupServer(...tusUploadHandlers, ...handlers);
