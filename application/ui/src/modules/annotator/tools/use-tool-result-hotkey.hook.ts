// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useHotkeys } from 'react-hotkeys-hook';

// Tool option bars contain inputs too (sliders, switches), but text fields elsewhere must keep Enter/Escape.
const isTextEntry = ({ target }: KeyboardEvent) =>
    target instanceof HTMLElement &&
    (target.isContentEditable || target.matches('textarea, input:not([type=range]):not([type=checkbox])'));

// Capture phase + stopPropagation so Enter doesn't also submit the media item and Escape doesn't close the annotator.
export const useToolResultHotkey = (hotkey: string, callback: () => void, enabled: boolean) => {
    useHotkeys(
        hotkey,
        (event) => {
            event.stopPropagation();
            callback();
        },
        {
            enabled,
            preventDefault: true,
            enableOnFormTags: true,
            ignoreEventWhen: isTextEntry,
            eventListenerOptions: { capture: true },
        },
        [callback]
    );
};
