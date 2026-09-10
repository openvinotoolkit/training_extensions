// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Label } from '@/api/types';
import { ActionButton, Flex, PressableElement, Text, Tooltip, TooltipTrigger } from '@geti-ui/ui';
import { Cross } from '@geti-ui/ui/icons';

import { formatHotkeyForDisplay } from '../../../../shared/hotkeys-definition';

import styles from './label-tag.module.scss';

type LabelTagProps = {
    label: Label;
    onDelete: (id: string) => void;
};

const LabelTagContent = ({ label, onDelete }: LabelTagProps) => {
    return (
        <Flex
            alignItems={'center'}
            gap={'size-100'}
            UNSAFE_style={{ '--labelColor': label.color }}
            UNSAFE_className={styles.labelTag}
        >
            <Text>{label.name}</Text>
            <ActionButton
                isQuiet
                onPress={() => onDelete(label.id)}
                aria-label={`Delete label ${label.name}`}
                UNSAFE_className={styles.deleteLabel}
            >
                <Cross />
            </ActionButton>
        </Flex>
    );
};

export const LabelTag = ({ label, onDelete }: LabelTagProps) => {
    if (label.hotkey != null) {
        return (
            <TooltipTrigger>
                <PressableElement>
                    <LabelTagContent label={label} onDelete={onDelete} />
                </PressableElement>
                <Tooltip>
                    <Text>{formatHotkeyForDisplay(label.hotkey)}</Text>
                </Tooltip>
            </TooltipTrigger>
        );
    }

    return <LabelTagContent label={label} onDelete={onDelete} />;
};
