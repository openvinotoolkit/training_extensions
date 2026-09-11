// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Label } from '@/api/types';
import type { TranslateFn } from '@/i18n';

import { convertHotkeyToOSFormat } from '../../shared/hotkeys-definition';

export const validateLabelName = (
    name: string,
    existingLabels: Label[],
    t: TranslateFn,
    excludeId?: string
): string | undefined => {
    const trimmedName = name.trim();

    const isDuplicate = existingLabels.some((label) => label.name === trimmedName && label.id !== excludeId);

    if (isDuplicate) {
        return t('labels.fields.nameExists');
    }

    return undefined;
};

export const validateLabelHotkey = (hotkey: string, allHotkeys: string[], t: TranslateFn): string | undefined => {
    const osFormatHotkeys = allHotkeys
        // Some hotkeys bind several keys at once, e.g. 'backspace, delete'
        .flatMap((key) => key.split(','))
        .map((key) => convertHotkeyToOSFormat(key.trim()).toLowerCase());

    if (osFormatHotkeys.includes(hotkey.toLowerCase())) {
        return t('labels.fields.hotkeyExists');
    }

    return undefined;
};
