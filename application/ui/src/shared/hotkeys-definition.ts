// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { TaskType } from '@/api/types';
import { isMac } from '@react-aria/utils';

const CTRL_KEY = 'ctrl';
const COMMAND_KEY = 'meta';
const CTRL_OR_COMMAND_KEY = isMac() ? COMMAND_KEY : CTRL_KEY;

export const HOTKEYS = {
    undo: `${CTRL_OR_COMMAND_KEY}+z`,
    redo: `${CTRL_OR_COMMAND_KEY}+y`,
    redoAlt: `${CTRL_OR_COMMAND_KEY}+shift+z`,
    toggleAnnotationsVisibility: 'a',
    deleteAnnotation: 'delete',
    deleteAnnotationAlternative: 'backspace',
    fitToScreen: 'r',
    selectionTool: 'v',
    boundingBoxTool: 'b',
    autoSegmentation: 's',
    polygonTool: 'p',
    magneticLassoTool: 'm',
    selectAllAnnotations: `${CTRL_OR_COMMAND_KEY}+a`,
    deselectAllAnnotations: `${CTRL_OR_COMMAND_KEY}+d`,
    selectNextAnnotation: 'tab',
    submitAlternative: 'enter',
    submit: `${CTRL_OR_COMMAND_KEY}+s`,
    previousMedia: 'arrowup, arrowleft',
    nextMedia: 'arrowdown, arrowright',
} as const;

const COMMON_HOTKEYS = {
    undo: HOTKEYS.undo,
    redo: HOTKEYS.redo,
    redoAlt: HOTKEYS.redoAlt,
    toggleAnnotationsVisibility: HOTKEYS.toggleAnnotationsVisibility,
    deleteAnnotation: HOTKEYS.deleteAnnotation,
    fitToScreen: HOTKEYS.fitToScreen,
    selectAllAnnotations: HOTKEYS.selectAllAnnotations,
    deselectAllAnnotations: HOTKEYS.deselectAllAnnotations,
    selectNextAnnotation: HOTKEYS.selectNextAnnotation,
    submitAlternative: HOTKEYS.submitAlternative,
    submit: HOTKEYS.submit,
    previousMedia: HOTKEYS.previousMedia,
    nextMedia: HOTKEYS.nextMedia,
} as const;

const SELECTION_TOOL_HOTKEY = {
    selectionTool: HOTKEYS.selectionTool,
} as const;

const AUTO_SEGMENTATION_HOTKEY = {
    autoSegmentation: HOTKEYS.autoSegmentation,
} as const;

export const TASK_HOTKEYS = {
    detection: {
        boundingBoxTool: HOTKEYS.boundingBoxTool,
        polygonTool: HOTKEYS.polygonTool,
        magneticLassoTool: HOTKEYS.magneticLassoTool,
        ...COMMON_HOTKEYS,
        ...SELECTION_TOOL_HOTKEY,
        ...AUTO_SEGMENTATION_HOTKEY,
    },
    instance_segmentation: {
        polygonTool: HOTKEYS.polygonTool,
        magneticLassoTool: HOTKEYS.magneticLassoTool,
        ...COMMON_HOTKEYS,
        ...SELECTION_TOOL_HOTKEY,
        ...AUTO_SEGMENTATION_HOTKEY,
    },
    classification: {
        ...COMMON_HOTKEYS,
    },
} satisfies Record<TaskType, Record<string, string>>;

/**
 * Converts a hotkey string to a user-friendly display format.
 * Replaces 'meta' with 'cmd' and converts to uppercase.
 *
 * @param key - The hotkey string (e.g., 'meta+z' or 'ctrl+z')
 * @returns The formatted hotkey string (e.g., 'CMD+Z' or 'CTRL+Z')
 *
 * @example
 * ```typescript
 * formatHotkeyForDisplay('meta+z') // Returns 'CMD+Z' on macOS
 * formatHotkeyForDisplay('ctrl+z') // Returns 'CTRL+Z' on Windows/Linux
 * formatHotkeyForDisplay('a') // Returns 'A'
 * ```
 */
export const formatHotkeyForDisplay = (key: string): string => {
    if (key.includes(COMMAND_KEY)) {
        return key.replace(COMMAND_KEY, 'cmd').toLocaleUpperCase();
    }

    return key.toLocaleUpperCase();
};

/**
 * Converts a hotkey string to the appropriate format for the current operating system.
 * Replaces 'ctrl' with 'meta' on macOS, or 'meta' with 'ctrl' on Windows/Linux.
 *
 * @param key - The hotkey string to convert (e.g., 'ctrl+z' or 'meta+z')
 * @returns The OS-specific hotkey string (e.g., 'meta+z' on macOS or 'ctrl+z' on Windows/Linux)
 *
 * @example
 * ```typescript
 * // On macOS:
 * convertHotkeyToOSFormat('ctrl+z') // Returns 'meta+z'
 * convertHotkeyToOSFormat('meta+z') // Returns 'meta+z'
 *
 * // On Windows/Linux:
 * convertHotkeyToOSFormat('meta+z') // Returns 'ctrl+z'
 * convertHotkeyToOSFormat('ctrl+z') // Returns 'ctrl+z'
 * ```
 */
export const convertHotkeyToOSFormat = (key: string): string => {
    if (isMac() && key.includes(CTRL_KEY)) {
        return key.replace(CTRL_KEY, COMMAND_KEY);
    }

    if (!isMac() && key.includes(COMMAND_KEY)) {
        return key.replace(COMMAND_KEY, CTRL_KEY);
    }

    return key;
};
