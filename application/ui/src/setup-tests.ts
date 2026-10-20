// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import * as jestDomMatchers from '@testing-library/jest-dom/matchers';
import fetchPolyfill, { Request as RequestPolyfill } from 'node-fetch';

import { server } from './msw-node-setup';

import './test-utils/mock-event-source';

expect.extend(jestDomMatchers);

beforeAll(() => {
    server.listen({ onUnhandledRequest: 'error' });
});

afterEach(() => {
    server.resetHandlers();
});

afterAll(() => {
    server.close();
});

// Why we need these polyfills:
// https://github.com/reduxjs/redux-toolkit/issues/4966#issuecomment-3115230061
Object.defineProperty(global, 'fetch', {
    // MSW will overwrite this to intercept requests
    writable: true,
    value: fetchPolyfill,
});

Object.defineProperty(global, 'Request', {
    writable: false,
    value: RequestPolyfill,
});

// For downloading logs and model files, we use URL.createObjectURL which is not
// implemented in jsdom, so we need to mock it.
Object.defineProperty(URL, 'createObjectURL', {
    writable: true,
    value: rs.fn(() => 'blob:mock-url'),
});

Object.defineProperty(URL, 'revokeObjectURL', {
    writable: true,
    value: rs.fn(),
});

// Mock ResizeObserver which is not available in jsdom
class ResizeObserverMock {
    observe = rs.fn();
    unobserve = rs.fn();
    disconnect = rs.fn();
}

global.ResizeObserver = ResizeObserverMock;

// Pointer capture is not implemented in jsdom, and sonner's swipe-to-dismiss handler relies on it
Object.defineProperty(Element.prototype, 'setPointerCapture', { writable: true, value: rs.fn() });
Object.defineProperty(Element.prototype, 'releasePointerCapture', { writable: true, value: rs.fn() });
Object.defineProperty(Element.prototype, 'hasPointerCapture', { writable: true, value: rs.fn(() => false) });

class IntersectionObserverMock {
    constructor(_callback: IntersectionObserverCallback, _options?: IntersectionObserverInit) {}

    observe = rs.fn();
    unobserve = rs.fn();
    disconnect = rs.fn();
    takeRecords = rs.fn(() => []);
}

global.IntersectionObserver = IntersectionObserverMock as unknown as typeof IntersectionObserver;

class ImageDataMock {
    data: Uint8ClampedArray;
    width: number;
    height: number;

    constructor(data: Uint8ClampedArray, width: number, height?: number) {
        this.data = data;
        this.width = width;
        this.height = height ?? 1;
    }
}

if (typeof global.ImageData === 'undefined') {
    global.ImageData = ImageDataMock as unknown as typeof ImageData;
}

const createLocalStorageMock = () => {
    let store: Record<string, string> = {};

    return {
        getItem: (key: string) => store[key] || null,
        setItem: (key: string, value: string) => {
            store[key] = value.toString();
        },
        removeItem: (key: string) => {
            delete store[key];
        },
        clear: () => {
            store = {};
        },
        get length() {
            return Object.keys(store).length;
        },
        key: (index: number) => {
            const keys = Object.keys(store);
            return keys[index] || null;
        },
    };
};

rs.stubGlobal('localStorage', createLocalStorageMock());
