// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { executeWithTimeout } from './execute-with-timeout';

describe('executeWithTimeout', () => {
    beforeEach(() => {
        rs.useFakeTimers();
    });

    afterEach(() => {
        rs.useRealTimers();
    });

    it('returns resolved value before timeout', async () => {
        const promise = executeWithTimeout(Promise.resolve('ok'), 'SAM decoder', 1000);

        await expect(promise).resolves.toBe('ok');
    });

    it('rejects when operation exceeds timeout', async () => {
        const neverResolvingPromise = new Promise<string>(() => undefined);
        const promise = executeWithTimeout(neverResolvingPromise, 'SAM decoder', 10);
        const rejectionExpectation = expect(promise).rejects.toThrow(/SAM decoder timed out after 10ms/i);

        await rs.advanceTimersByTimeAsync(11);

        await rejectionExpectation;
    });
});
