// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { FocusEvent, KeyboardEvent, useRef, useState } from 'react';

import { HotkeyField } from '@/components/label-fields/hotkey-field.component';
import { LabelColorPicker } from '@/components/label-fields/label-color-picker.component';
import { ActionButton, DOMRefValue, Flex, Grid, TextField, useUnwrapDOMRef, View } from '@geti-ui/ui';
import { Add, Close } from '@geti-ui/ui/icons';

import { getRandomDistinctColor } from '../../label-utils';

import classes from '../label-row/label-row.module.scss';

type NewLabelRowProps = {
    onSave: (name: string, color: string, hotkey?: string) => void;
    onCancel: () => void;
    validateName: (name: string, excludeId?: string) => string | undefined;
    validateHotkey: (newHotkey: string, excludeId?: string) => string | undefined;
};

export const NewLabelRow = ({ onSave, onCancel, validateName, validateHotkey }: NewLabelRowProps) => {
    const rowRef = useRef<DOMRefValue<HTMLDivElement>>(null);
    const rowRefUnwrapped = useUnwrapDOMRef(rowRef);
    const [name, setName] = useState('');
    const [hotkey, setHotkey] = useState('');
    const [color, setColor] = useState(getRandomDistinctColor);

    const isEmptyName = name.trim().length === 0;
    const validationError = isEmptyName ? undefined : validateName(name);

    const canSave = (newName: string) => {
        const trimmedName = newName.trim();

        return trimmedName.length > 0 && validateName(trimmedName) === undefined;
    };

    const isCreateButtonDisabled = !canSave(name);

    const handleSave = () => {
        if (canSave(name)) {
            onSave(name.trim(), color, hotkey.trim() === '' ? undefined : hotkey.trim());
        }
    };

    const handleNameKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
        if (event.key === 'Enter') {
            handleSave();
        } else if (event.key === 'Escape') {
            onCancel();
        }
    };

    const handleHotkeyChange = (newHotkey: string | null) => {
        setHotkey(newHotkey ?? '');
    };

    const handleBlur = (event: FocusEvent<HTMLInputElement>) => {
        // Check if the blur target is within the row (e.g., clicking color picker)
        const relatedTarget = event.relatedTarget as Node | null;
        if (relatedTarget && rowRefUnwrapped.current?.contains(relatedTarget)) {
            return;
        }

        const trimmedName = name.trim();

        if (canSave(name)) {
            onSave(trimmedName, color);
        } else if (trimmedName.length === 0) {
            onCancel();
        }
    };

    const handleHotkeyUpdate = () => {
        if (validateHotkey(hotkey) !== undefined) {
            return;
        }

        handleSave();
    };

    return (
        <Grid
            ref={rowRef}
            columns={['size-350', 'size-400', '1fr', 'size-400', 'size-400']}
            gap={'size-100'}
            alignItems={'start'}
            UNSAFE_className={classes.labelRow}
            UNSAFE_style={{ '--label-color': color }}
        >
            <View />

            <LabelColorPicker color={color} onChange={setColor} />

            <Flex gap={'size-100'}>
                <TextField
                    // eslint-disable-next-line jsx-a11y/no-autofocus
                    autoFocus
                    aria-label={'New label name'}
                    placeholder={'Label name'}
                    value={name}
                    onChange={setName}
                    onKeyDown={handleNameKeyDown}
                    onBlur={handleBlur}
                    width={'100%'}
                    errorMessage={validationError}
                    validationState={validationError ? 'invalid' : undefined}
                />

                <HotkeyField
                    hotkey={hotkey}
                    onEnter={handleHotkeyUpdate}
                    onHotkeyChange={handleHotkeyChange}
                    aria-label={'New label hotkey'}
                    errorMessage={validateHotkey(hotkey)}
                />
            </Flex>

            <ActionButton
                isQuiet
                aria-label={'Create new label'}
                onPress={handleSave}
                isDisabled={isCreateButtonDisabled}
            >
                <Add />
            </ActionButton>

            <ActionButton
                aria-label='Cancel new label'
                isQuiet
                onPress={onCancel}
                UNSAFE_className={classes.deleteButton}
            >
                <Close />
            </ActionButton>
        </Grid>
    );
};
