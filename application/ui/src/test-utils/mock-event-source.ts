// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

type MockEventSource = {
    onopen: ((event: Event) => void) | null;
    onmessage: ((event: MessageEvent) => void) | null;
    onerror: ((event: Event) => void) | null;
    close: ReturnType<typeof rs.fn>;
    url: string;
    readyState: number;
};

let mockEventSourceInstances: MockEventSource[] = [];

export const MockEventSourceConstructor = rs.fn().mockImplementation(function (url: string) {
    const instance: MockEventSource = {
        onopen: null,
        onmessage: null,
        onerror: null,
        close: rs.fn(),
        url,
        readyState: 0,
    };
    mockEventSourceInstances.push(instance);
    return instance;
});

export const getLastEventSource = (): MockEventSource => {
    const instance = mockEventSourceInstances.at(-1);
    if (!instance) {
        throw new Error('No MockEventSource instance found');
    }
    return instance;
};

export const simulateSSEMessage = (instance: MockEventSource, data: unknown) => {
    instance.onmessage?.({ data: JSON.stringify(data) } as MessageEvent);
};

export const simulateSSEOpen = (instance: MockEventSource) => {
    instance.onopen?.({} as Event);
};

export const simulateSSEError = (instance: MockEventSource) => {
    instance.onerror?.({} as Event);
};

export const resetMockEventSource = () => {
    mockEventSourceInstances = [];
    MockEventSourceConstructor.mockClear();
};

rs.stubGlobal('EventSource', MockEventSourceConstructor);
