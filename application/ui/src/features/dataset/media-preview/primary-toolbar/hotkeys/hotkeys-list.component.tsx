// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { Divider, Grid, Keyboard, Text } from '@geti-ui/ui';

import { formatHotkeyForDisplay, HOTKEYS } from '../../../../../shared/hotkeys-definition';
import { useAvailableTools } from '../../../../annotator/tools/annotator-tools/use-available-tools';

interface HotkeyItemProps {
    hotkeyName: string;
    hotkey: string;
}

const HotkeyItem = ({ hotkeyName, hotkey }: HotkeyItemProps) => {
    return (
        <>
            <Text>{hotkeyName}</Text>
            <Keyboard>{hotkey}</Keyboard>
        </>
    );
};

export const HotkeysList = () => {
    const { t } = useTranslation();
    const availableTools = useAvailableTools();
    const submitHotkey = formatHotkeyForDisplay(HOTKEYS.submit);
    const submitAlternativeHotkey = formatHotkeyForDisplay(HOTKEYS.submitAlternative);

    return (
        <Grid columns={['2fr', '1fr']} rowGap={'size-100'}>
            <HotkeyItem
                hotkeyName={t('annotator.hotkeys.submitAnnotationsPredictions')}
                hotkey={`${submitHotkey} or ${submitAlternativeHotkey}`}
            />
            <Divider size='S' gridColumn={'1/-1'} />
            <HotkeyItem
                hotkeyName={t('annotator.hotkeys.previousMedia')}
                hotkey={formatHotkeyForDisplay(HOTKEYS.previousMedia)}
            />
            <HotkeyItem
                hotkeyName={t('annotator.hotkeys.nextMedia')}
                hotkey={formatHotkeyForDisplay(HOTKEYS.nextMedia)}
            />
            <Divider size='S' gridColumn={'1/-1'} />
            {availableTools.map((tool) => (
                <HotkeyItem key={tool.type} hotkeyName={tool.label} hotkey={formatHotkeyForDisplay(tool.hotkey)} />
            ))}
            <Divider size='S' gridColumn={'1/-1'} />
            <HotkeyItem hotkeyName={t('annotator.hotkeys.undo')} hotkey={formatHotkeyForDisplay(HOTKEYS.undo)} />
            <HotkeyItem
                hotkeyName={t('annotator.hotkeys.redo')}
                hotkey={`${formatHotkeyForDisplay(HOTKEYS.redo)} or ${formatHotkeyForDisplay(HOTKEYS.redoAlt)}`}
            />
            <HotkeyItem
                hotkeyName={t('annotator.hotkeys.deleteSelectedAnnotation')}
                hotkey={formatHotkeyForDisplay(HOTKEYS.delete)}
            />
            <HotkeyItem
                hotkeyName={t('annotator.hotkeys.toggleAnnotationsVisibility')}
                hotkey={formatHotkeyForDisplay(HOTKEYS.toggleAnnotationsVisibility)}
            />
            <HotkeyItem
                hotkeyName={t('annotator.hotkeys.selectAllAnnotations')}
                hotkey={formatHotkeyForDisplay(HOTKEYS.selectAllAnnotations)}
            />
            <HotkeyItem
                hotkeyName={t('annotator.hotkeys.deselectAllAnnotations')}
                hotkey={formatHotkeyForDisplay(HOTKEYS.deselectAllAnnotations)}
            />
            <HotkeyItem
                hotkeyName={t('annotator.hotkeys.selectNextAnnotation')}
                hotkey={formatHotkeyForDisplay(HOTKEYS.selectNextAnnotation)}
            />
            <Divider size='S' gridColumn={'1/-1'} />
            <HotkeyItem
                hotkeyName={t('annotator.hotkeys.resetZoom')}
                hotkey={formatHotkeyForDisplay(HOTKEYS.fitToScreen)}
            />
        </Grid>
    );
};
