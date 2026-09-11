// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { ActionButton, Flex, Tooltip, TooltipTrigger } from '@geti-ui/ui';
import { Redo, Undo } from '@geti-ui/ui/icons';
import { useHotkeys } from 'react-hotkeys-hook';

import { formatHotkeyForDisplay, HOTKEYS } from '../../../../../shared/hotkeys-definition';
import { useUndoRedo } from './undo-redo-provider.component';

export const UndoRedo = ({ isDisabled }: { isDisabled?: boolean }) => {
    const { t } = useTranslation();
    const { undo, canUndo, redo, canRedo } = useUndoRedo();

    useHotkeys(HOTKEYS.undo, undo, { enabled: canUndo, preventDefault: true }, [undo, canUndo]);

    useHotkeys(`${HOTKEYS.redo}, ${HOTKEYS.redoAlt}`, redo, { enabled: canRedo, preventDefault: true }, [
        redo,
        canRedo,
    ]);

    const undoHotkey = formatHotkeyForDisplay(HOTKEYS.undo);
    const redoHotkeys = `${formatHotkeyForDisplay(HOTKEYS.redo)} or ${formatHotkeyForDisplay(HOTKEYS.redoAlt)}`;

    const undoAriaLabel = `Undo (${undoHotkey})`;
    const redoAriaLabel = `Redo (${redoHotkeys})`;
    const undoLabel = t('annotator.actions.undo', { hotkey: undoHotkey });
    const redoLabel = t('annotator.actions.redo', { hotkeys: redoHotkeys });

    return (
        <Flex alignItems='center' direction={'column'} justifyContent={'center'} data-testid='undo-redo-tools'>
            <TooltipTrigger placement={'end'}>
                <ActionButton
                    isQuiet
                    id='undo-button'
                    data-testid='undo-button'
                    onPress={undo}
                    aria-label={undoAriaLabel}
                    isDisabled={!canUndo || isDisabled}
                >
                    <Undo />
                </ActionButton>
                <Tooltip>{undoLabel}</Tooltip>
            </TooltipTrigger>

            <TooltipTrigger placement={'end'}>
                <ActionButton
                    isQuiet
                    id='redo-button'
                    data-testid='redo-button'
                    aria-label={redoAriaLabel}
                    onPress={redo}
                    isDisabled={!canRedo || isDisabled}
                >
                    <Redo />
                </ActionButton>
                <Tooltip>{redoLabel}</Tooltip>
            </TooltipTrigger>
        </Flex>
    );
};
