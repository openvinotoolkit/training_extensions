// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { IconWrapper } from '@/components/icon-wrapper/icon-wrapper.component';
import { useSetZoom } from '@/components/zoom/zoom.provider';
import { ActionButton, Tooltip, TooltipTrigger } from '@geti-ui/ui';
import { FitScreen } from '@geti-ui/ui/icons';
import { useHotkeys } from 'react-hotkeys-hook';

import { formatHotkeyForDisplay, HOTKEYS } from '../../../../../shared/hotkeys-definition';

export const ZoomFitScreen = () => {
    const { fitToScreen } = useSetZoom();

    useHotkeys(HOTKEYS.fitToScreen, fitToScreen, [fitToScreen]);

    const label = `Fit to screen (${formatHotkeyForDisplay(HOTKEYS.fitToScreen)})`;

    return (
        <TooltipTrigger>
            <ActionButton isQuiet onPress={fitToScreen} aria-label={label}>
                <IconWrapper>
                    <FitScreen />
                </IconWrapper>
            </ActionButton>
            <Tooltip>{label}</Tooltip>
        </TooltipTrigger>
    );
};
