// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { KeyboardEvent, useState } from 'react';

import type { Label } from '@/api/types';
import { HotkeyField } from '@/components/label-fields/hotkey-field.component';
import { LabelColorPicker } from '@/components/label-fields/label-color-picker.component';
import { SilentCheckbox } from '@/components/label-fields/silent-checkbox.component';
import { ActionButton, Flex, Grid, TextField, Tooltip, TooltipTrigger } from '@geti-ui/ui';
import { Delete, Pin, Unpin } from '@geti-ui/ui/icons';

import { useDebounce } from '../../../../hooks/use-debounce.hook';
import { isNonEmptyString } from '../../../../shared/util';

import classes from './label-row.module.scss';

const COLOR_DEBOUNCE_MS = 300;

export type LabelRowProps = {
    label: Label;
    isSelected: boolean;
    isPinned: boolean;
    onSelect: () => void;
    onDelete: (label: Label) => void;
    onTogglePin: (label: Label) => void;
    onUpdate: (labelId: string, updates: { name: string; color: string; hotkey: string | null | undefined }) => void;
    validateName: (name: string, excludeId?: string) => string | undefined;
    validateHotkey: (newHotkey: string, excludeId?: string) => string | undefined;
};

const onEnter = (handler: () => void) => (event: KeyboardEvent) => {
    event.key === 'Enter' && handler();
};

export const LabelRow = ({
    label,
    isSelected,
    isPinned,
    onSelect,
    onDelete,
    onTogglePin,
    onUpdate,
    validateName,
    validateHotkey,
}: LabelRowProps) => {
    const [name, setName] = useState(label.name);
    const [color, setColor] = useState(label.color);
    const [hotkey, setHotkey] = useState(label.hotkey ?? '');

    const validationError = validateName(name, label.id);
    const hotkeyValidationError = validateHotkey(hotkey, label.id);
    const trimmedHotkey = isNonEmptyString(hotkey) ? hotkey.trim() : undefined;
    const hasValidationErrors =
        validationError !== undefined || hotkeyValidationError !== undefined || name.trim() === '';

    const getValidName = () => (validationError || name.trim() === '' ? label.name : name.trim());

    const handleUpdateName = () => {
        const isNameUnchanged = name.trim() === label.name;
        if (hasValidationErrors || isNameUnchanged) return;

        onUpdate(label.id, { name: name.trim(), color, hotkey: trimmedHotkey });
    };

    const handleHotkeyChange = () => {
        const isHotkeyChanged = label.hotkey !== trimmedHotkey;
        if (hasValidationErrors || !isHotkeyChanged) return;

        onUpdate(label.id, { name: name.trim(), color, hotkey: trimmedHotkey });
    };

    const debouncedUpdate = useDebounce(
        (newColor: string, currentName: string) => {
            onUpdate(label.id, { name: currentName, color: newColor, hotkey: trimmedHotkey });
        },
        COLOR_DEBOUNCE_MS,
        [onUpdate, label.id, hotkey]
    );

    const handleColorChange = (newColor: string) => {
        setColor(newColor);
        debouncedUpdate(newColor, getValidName());
    };

    return (
        <Grid
            columns={['size-350', 'size-400', '1fr', 'size-400', 'size-400']}
            gap={'size-100'}
            alignItems={'start'}
            UNSAFE_className={classes.labelRow}
            UNSAFE_style={{ '--label-color': color }}
        >
            <SilentCheckbox isSelected={isSelected} onChange={onSelect} aria-label={`Select ${label.name} label`} />

            <LabelColorPicker color={color} onChange={handleColorChange} />

            <Flex gap={'size-100'}>
                <TextField
                    width={'100%'}
                    value={name}
                    aria-label={'Label name'}
                    placeholder={'Label name'}
                    onChange={setName}
                    onBlur={handleUpdateName}
                    onKeyDown={onEnter(handleUpdateName)}
                    errorMessage={validationError}
                    validationState={validationError ? 'invalid' : undefined}
                />

                <HotkeyField
                    hotkey={hotkey}
                    onHotkeyChange={(newHotkey) => setHotkey(newHotkey ?? '')}
                    onEnter={handleHotkeyChange}
                    onBlur={handleHotkeyChange}
                    aria-label={'Edited hotkey'}
                    errorMessage={hotkeyValidationError}
                />
            </Flex>

            <TooltipTrigger>
                <ActionButton
                    aria-label={isPinned ? `Unpin ${label.name} label` : `Pin ${label.name} label`}
                    isQuiet
                    UNSAFE_className={isPinned ? classes.pinButtonPinned : classes.pinButton}
                    onPress={() => onTogglePin(label)}
                >
                    {isPinned ? <Pin /> : <Unpin />}
                </ActionButton>
                <Tooltip>{isPinned ? 'Unpin label' : 'Pin label'}</Tooltip>
            </TooltipTrigger>

            <ActionButton
                aria-label={`Delete ${label.name} label`}
                isQuiet
                UNSAFE_className={classes.deleteButton}
                onPress={() => onDelete(label)}
            >
                <Delete />
            </ActionButton>
        </Grid>
    );
};
