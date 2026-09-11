// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { ActionButton, Tooltip, TooltipTrigger } from '@geti-ui/ui';
import { Invisible, Visible } from '@geti-ui/ui/icons';
import { useHotkeys } from 'react-hotkeys-hook';

import { useAnnotationVisibility } from '../../../../shared/annotator/annotation-visibility-provider.component';
import { formatHotkeyForDisplay, HOTKEYS } from '../../../../shared/hotkeys-definition';

export const ToggleAnnotationsVisibility = () => {
    const { t } = useTranslation();
    const { isVisible, toggleVisibility } = useAnnotationVisibility();

    useHotkeys(HOTKEYS.toggleAnnotationsVisibility, toggleVisibility, [toggleVisibility]);

    const hotkey = formatHotkeyForDisplay(HOTKEYS.toggleAnnotationsVisibility);
    const ariaLabel = `${isVisible ? 'Hide' : 'Show'} annotations (${hotkey})`;
    const label = isVisible
        ? t('annotator.actions.hideAnnotations', { hotkey })
        : t('annotator.actions.showAnnotations', { hotkey });

    return (
        <TooltipTrigger placement={'right'}>
            <ActionButton aria-label={ariaLabel} isQuiet onPress={toggleVisibility}>
                {isVisible ? <Visible /> : <Invisible />}
            </ActionButton>
            <Tooltip>{label}</Tooltip>
        </TooltipTrigger>
    );
};
