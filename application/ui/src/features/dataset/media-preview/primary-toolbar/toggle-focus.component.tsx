// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { IconWrapper } from '@/components/icon-wrapper/icon-wrapper.component';
import { ActionButton, Tooltip, TooltipTrigger } from '@geti-ui/ui';
import { LabelGroup } from '@geti-ui/ui/icons';

import { useAnnotationVisibility } from '../../../../shared/annotator/annotation-visibility-provider.component';

export const ToggleFocus = () => {
    const { toggleFocus, isFocussed } = useAnnotationVisibility();

    return (
        <TooltipTrigger>
            <ActionButton isQuiet onPress={toggleFocus} aria-label={'Toggle focus'}>
                <IconWrapper isSelected={isFocussed}>
                    <LabelGroup />
                </IconWrapper>
            </ActionButton>
            <Tooltip>Toggle focus</Tooltip>
        </TooltipTrigger>
    );
};
