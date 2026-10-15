// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { Button, Flex, Text } from '@geti-ui/ui';
import { useHotkeys } from 'react-hotkeys-hook';

import { HOTKEYS } from '../../../../shared/hotkeys-definition';
import { Toolbar } from '../../shell/toolbar-container/toolbar-container.component';
import { useSSIMState } from './ssim-state-provider.component';

// Capture phase + stopPropagation so Enter doesn't also submit the media item and Escape doesn't close the annotator.
const useToolResultHotkey = (hotkey: string, callback: () => void, enabled: boolean) => {
    useHotkeys(
        hotkey,
        (event) => {
            event.stopPropagation();
            callback();
        },
        { enabled, preventDefault: true, eventListenerOptions: { capture: true } },
        [callback]
    );
};

export const SSIMOptions = () => {
    const { t } = useTranslation();
    const { hasPreview, isProcessing, accept, reject } = useSSIMState();
    const canAccept = hasPreview && !isProcessing;
    const canReject = hasPreview || isProcessing;

    useToolResultHotkey(HOTKEYS.acceptToolResult, accept, canAccept);
    useToolResultHotkey(HOTKEYS.rejectToolResult, reject, canReject);

    return (
        <Toolbar.Container>
            <Toolbar.Section>
                <Flex alignItems={'center'} gap={'size-100'}>
                    <Text>{t('annotator.tools.ssim.label')}</Text>
                    <Button variant={'accent'} onPress={accept} isDisabled={!canAccept}>
                        {t('annotator.tools.ssim.options.accept')}
                    </Button>
                    <Button variant={'secondary'} onPress={reject} isDisabled={!canReject}>
                        {t('annotator.tools.ssim.options.reject')}
                    </Button>
                </Flex>
            </Toolbar.Section>
        </Toolbar.Container>
    );
};
