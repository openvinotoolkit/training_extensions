// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useRef, useState } from 'react';

import type { Label, TaskType } from '@/api/types';
import { HotkeyField } from '@/components/label-fields/hotkey-field.component';
import { LabelColorPicker } from '@/components/label-fields/label-color-picker.component';
import { validateLabelHotkey, validateLabelName } from '@/components/label-fields/label-validation';
import { ActionButton, DOMRefValue, Grid, TextField, TextFieldRef, useUnwrapDOMRef, View } from '@geti-ui/ui';
import { Add } from '@geti-ui/ui/icons';
import { useEventListener } from 'hooks/event-listener.hook';
import { v4 as uuid } from 'uuid';

import { TASK_HOTKEYS } from '../../../../shared/hotkeys-definition';
import { getRandomDistinctColor } from '../../../annotator/label-utils';

const getInitialLabel = (): Label => ({ id: uuid(), color: getRandomDistinctColor(), name: '', hotkey: null });

type CreateLabelProps = {
    onCreate: (label: Label) => void;
    labels: Label[];
    taskType: TaskType;
};

export const CreateLabel = ({ labels, onCreate, taskType }: CreateLabelProps) => {
    const [newLabel, setNewLabel] = useState<Label>(getInitialLabel);
    const containerRef = useRef<DOMRefValue<HTMLDivElement>>(null);
    const inputRef = useRef<TextFieldRef<HTMLInputElement>>(null);
    const inputRefUnwrapped = useUnwrapDOMRef(inputRef);

    const labelsHotkeys = labels.map((label) => label.hotkey).filter((hotkey) => hotkey != null);
    const appHotkeys = Object.values(TASK_HOTKEYS[taskType]);
    const allHotkeys = [...labelsHotkeys, ...appHotkeys];

    const validationResult = validateLabelName(newLabel.name, labels);
    const hotkeyError = newLabel.hotkey ? validateLabelHotkey(newLabel.hotkey, allHotkeys) : undefined;
    const isCreateLabelDisabled = newLabel.name.trim().length === 0 || validationResult !== undefined;

    const createLabel = () => {
        if (isCreateLabelDisabled) return;

        onCreate({ ...newLabel, name: newLabel.name.trim() });
        setNewLabel(getInitialLabel);
    };

    useEventListener(
        'keydown',
        (event) => {
            if (event.key === 'Enter') {
                event.preventDefault();
                event.stopImmediatePropagation();

                createLabel();

                inputRef.current?.focus();
            }
        },
        inputRefUnwrapped
    );

    return (
        <Grid
            columns={['size-400', '1fr', 'size-1600', 'size-400']}
            gap={'size-50'}
            maxWidth={'640px'}
            width={'100%'}
            alignItems={'start'}
            ref={containerRef}
        >
            <LabelColorPicker
                onChange={(newColor) => {
                    setNewLabel((prevLabel) => ({ ...prevLabel, color: newColor }));
                }}
                color={newLabel.color}
            />
            <View>
                <TextField
                    ref={inputRef}
                    aria-label={'Create label input'}
                    placeholder={'Create label'}
                    value={newLabel.name}
                    onChange={(newName) => setNewLabel((prevLabel) => ({ ...prevLabel, name: newName }))}
                    errorMessage={validationResult}
                    validationState={validationResult === undefined ? undefined : 'invalid'}
                    width={'100%'}
                />
            </View>
            <View>
                <HotkeyField
                    hotkey={newLabel.hotkey}
                    onHotkeyChange={(newHotkey) => setNewLabel((prevLabel) => ({ ...prevLabel, hotkey: newHotkey }))}
                    errorMessage={hotkeyError}
                />
            </View>

            <ActionButton
                isQuiet
                onPress={createLabel}
                isDisabled={isCreateLabelDisabled}
                aria-label={`Create label ${newLabel.name}`}
            >
                <Add />
            </ActionButton>
        </Grid>
    );
};
