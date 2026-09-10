// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { renderHook } from '@testing-library/react';

import { useClipboard } from './use-clipboard.hook';

const mockwriteText = vi.fn();
Object.assign(navigator, {
    clipboard: {
        writeText: mockwriteText,
    },
});

const mockedToast = vi.fn();
vi.mock('../../components/toast/toast.component', () => ({
    toast: (params: unknown) => mockedToast(params),
}));

describe('useClipboard', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('Copy and shows a info notification message', async () => {
        mockwriteText.mockResolvedValue(true);
        const textToCopy = 'this is a test';
        const confirmationMessage = 'copied';
        const { result } = renderHook(() => useClipboard());

        await result.current.copy(textToCopy, confirmationMessage, 'unused error message');

        expect(mockwriteText).toHaveBeenCalledWith(textToCopy);
        expect(mockedToast).toHaveBeenCalledWith({
            message: confirmationMessage,
            type: 'info',
        });
    });

    it('Copy and shows a error notification message', async () => {
        mockwriteText.mockRejectedValue(true);
        const textToCopy = 'this is a test';
        const errorMessage = 'error';
        const { result } = renderHook(() => useClipboard());

        await result.current.copy(textToCopy, '', errorMessage);

        expect(mockwriteText).toHaveBeenCalledWith(textToCopy);
        expect(mockedToast).toHaveBeenCalledWith({ message: errorMessage, type: 'error' });
    });
});
