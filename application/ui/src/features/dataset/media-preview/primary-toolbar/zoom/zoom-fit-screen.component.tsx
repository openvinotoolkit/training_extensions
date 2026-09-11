// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { IconWrapper } from '@/components/icon-wrapper/icon-wrapper.component';
import { useSetZoom } from '@/components/zoom/zoom.provider';
import { useTranslation } from '@/i18n';
import { ActionButton, Tooltip, TooltipTrigger } from '@geti-ui/ui';
import { FitScreen } from '@geti-ui/ui/icons';
import { useHotkeys } from 'react-hotkeys-hook';

import { formatHotkeyForDisplay, HOTKEYS } from '../../../../../shared/hotkeys-definition';

export const ZoomFitScreen = () => {
    const { t } = useTranslation();
    const { fitToScreen } = useSetZoom();

    useHotkeys(HOTKEYS.fitToScreen, fitToScreen, [fitToScreen]);

    const hotkey = formatHotkeyForDisplay(HOTKEYS.fitToScreen);
    const ariaLabel = `Fit to screen (${hotkey})`;
    const label = t('annotator.actions.fitToScreen', { hotkey });

    return (
        <TooltipTrigger>
            <ActionButton isQuiet onPress={fitToScreen} aria-label={ariaLabel}>
                <IconWrapper>
                    <FitScreen />
                </IconWrapper>
            </ActionButton>
            <Tooltip>{label}</Tooltip>
        </TooltipTrigger>
    );
};
