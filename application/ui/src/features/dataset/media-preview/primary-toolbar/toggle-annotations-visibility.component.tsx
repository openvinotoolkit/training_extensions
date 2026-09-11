// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ActionButton, Tooltip, TooltipTrigger } from '@geti-ui/ui';
import { Invisible, Visible } from '@geti-ui/ui/icons';
import { useHotkeys } from 'react-hotkeys-hook';

import { useAnnotationVisibility } from '../../../../shared/annotator/annotation-visibility-provider.component';
import { formatHotkeyForDisplay, HOTKEYS } from '../../../../shared/hotkeys-definition';

export const ToggleAnnotationsVisibility = () => {
    const { isVisible, toggleVisibility } = useAnnotationVisibility();

    useHotkeys(HOTKEYS.toggleAnnotationsVisibility, toggleVisibility, [toggleVisibility]);

    const hotkey = formatHotkeyForDisplay(HOTKEYS.toggleAnnotationsVisibility);
    const label = `${isVisible ? 'Hide' : 'Show'} annotations (${hotkey})`;

    return (
        <TooltipTrigger placement={'right'}>
            <ActionButton aria-label={label} isQuiet onPress={toggleVisibility}>
                {isVisible ? <Visible /> : <Invisible />}
            </ActionButton>
            <Tooltip>{label}</Tooltip>
        </TooltipTrigger>
    );
};
