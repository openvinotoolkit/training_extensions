// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { KeyboardEvent } from 'react';

import { useTranslation } from '@/i18n';
import { TextField } from '@geti-ui/ui';

import { formatHotkeyForDisplay } from '../../shared/hotkeys-definition';

type HotkeyFieldProps = {
    hotkey: string | null | undefined;
    onEnter?: () => void;
    onBlur?: () => void;
    onHotkeyChange: (hotkey: string | null) => void;
    errorMessage?: string;
};

const isEnter = (event: KeyboardEvent) => {
    return event.key === 'Enter';
};

const isBackspace = (event: KeyboardEvent) => {
    return event.key === 'Backspace';
};

const isTab = (event: KeyboardEvent) => {
    return event.key === 'Tab';
};

export const HotkeyField = ({ hotkey, errorMessage, onEnter, onHotkeyChange, onBlur }: HotkeyFieldProps) => {
    const { t } = useTranslation();
    const handleKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
        event.preventDefault();

        // We want to allow keyboard navigation with tabs
        if (isTab(event)) {
            return;
        }

        if (isBackspace(event)) {
            onHotkeyChange(null);

            return;
        }

        const { key, ctrlKey, altKey, shiftKey, metaKey } = event;

        // Ignore standalone modifier keys
        if (['Control', 'Alt', 'Shift', 'Meta', 'Enter'].includes(key)) {
            isEnter(event) && onEnter?.();

            return;
        }

        const modifiers: string[] = [];
        if (ctrlKey) modifiers.push('ctrl');
        if (metaKey) modifiers.push('meta');
        if (altKey) modifiers.push('alt');
        if (shiftKey) modifiers.push('shift');

        const hotkeyString = modifiers.length > 0 ? `${modifiers.join('+')}+${key}` : key;

        onHotkeyChange(hotkeyString);
    };

    const formattedHotkey = hotkey == null ? '' : formatHotkeyForDisplay(hotkey);

    return (
        <TextField
            aria-label={'Hotkey input'}
            placeholder={t('project.labels.hotkeyPlaceholder')}
            value={formattedHotkey}
            onKeyDown={handleKeyDown}
            onBlur={onBlur}
            width={'100%'}
            errorMessage={errorMessage}
            validationState={errorMessage ? 'invalid' : undefined}
        />
    );
};
