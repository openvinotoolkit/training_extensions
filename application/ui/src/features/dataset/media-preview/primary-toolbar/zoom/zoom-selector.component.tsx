// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { IconWrapper } from '@/components/icon-wrapper/icon-wrapper.component';
import { useSetZoom, useZoom } from '@/components/zoom/zoom.provider';
import { useTranslation } from '@/i18n';
import { ActionButton, Flex, Tooltip, TooltipTrigger } from '@geti-ui/ui';
import { Add, Remove } from '@geti-ui/ui/icons';

export const ZoomSelector = () => {
    const { t } = useTranslation();
    const zoom = useZoom();
    const { onZoomChange } = useSetZoom();

    return (
        <>
            <TooltipTrigger>
                <ActionButton
                    isQuiet
                    aria-label='Zoom In'
                    onPress={() => onZoomChange(1)}
                    isDisabled={zoom.scale >= zoom.maxZoomIn}
                >
                    <IconWrapper>
                        <Add />
                    </IconWrapper>
                </ActionButton>
                <Tooltip>{t('annotator.actions.zoomIn')}</Tooltip>
            </TooltipTrigger>

            <Flex justifyContent={'end'} width={'size-350'}>
                <span
                    aria-label={'Zoom level'}
                    data-value={zoom.scale}
                    style={{ fontSize: 'var(--spectrum-global-dimension-font-size-50)' }}
                >
                    {(zoom.scale * 100).toFixed(0)}%
                </span>
            </Flex>

            <TooltipTrigger>
                <ActionButton
                    isQuiet
                    aria-label='Zoom Out'
                    onPress={() => onZoomChange(-1)}
                    isDisabled={zoom.scale <= zoom.initialCoordinates.scale}
                >
                    <IconWrapper>
                        <Remove />
                    </IconWrapper>
                </ActionButton>
                <Tooltip>{t('annotator.actions.zoomOut')}</Tooltip>
            </TooltipTrigger>
        </>
    );
};
