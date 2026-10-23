// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useRef } from 'react';

export const useWithCancel = <Args extends unknown[], Result>(fn: (...args: Args) => Promise<Result>) => {
    const abortController = useRef<AbortController | null>(null);

    const cancellableCallback = async (...args: Args): Promise<Result> => {
        // Cancel any ongoing request
        abortController.current?.abort();

        // Create a new controller for THIS call and capture its signal in a local
        // variable BEFORE awaiting anything. Any future cancel() call will replace
        // abortController.current, but `controller.signal` here is a closed-over reference to
        // the abortController for THIS specific invocation and will correctly reflect
        // whether it was aborted — even after the ref has been replaced.
        const controller = new AbortController();
        abortController.current = controller;

        const result = await fn(...args);

        if (controller.signal.aborted) {
            throw new DOMException('Request aborted', 'AbortError');
        }

        return result;
    };

    const cancel = () => {
        abortController.current?.abort();
        abortController.current = null;
    };

    useEffect(() => {
        return () => {
            abortController.current?.abort();
        };
    }, []);

    return {
        call: cancellableCallback,
        cancel,
    };
};
