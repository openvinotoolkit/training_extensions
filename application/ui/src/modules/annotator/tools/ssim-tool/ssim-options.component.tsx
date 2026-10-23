// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { Button, Flex, Slider, Switch } from '@geti-ui/ui';

import { HOTKEYS } from '../../../../shared/hotkeys-definition';
import { Toolbar } from '../../shell/toolbar-container/toolbar-container.component';
import { useToolResultHotkey } from '../use-tool-result-hotkey.hook';
import { useSSIMState } from './ssim-state-provider.component';

export const SSIMOptions = () => {
    const { t } = useTranslation();
    const {
        hasPreview,
        isProcessing,
        accept,
        reject,
        toolState,
        setNumberOfItems,
        autoMergeDuplicates,
        setAutoMergeDuplicates,
    } = useSSIMState();
    const canReject = hasPreview || isProcessing;
    const numberOfMatches = toolState.matches.length;

    useToolResultHotkey(HOTKEYS.acceptToolResult, accept, hasPreview);
    useToolResultHotkey(HOTKEYS.rejectToolResult, reject, canReject);

    return (
        <Toolbar.Container>
            <Toolbar.Section>
                <Flex alignItems={'center'} gap={'size-200'}>
                    <Slider
                        label={t('annotator.tools.ssim.options.numberOfItems')}
                        labelPosition={'side'}
                        width={'size-3600'}
                        minValue={1}
                        maxValue={Math.max(1, numberOfMatches)}
                        step={1}
                        value={Math.max(1, toolState.threshold)}
                        onChange={setNumberOfItems}
                        isDisabled={!hasPreview || numberOfMatches <= 1}
                        isFilled
                    />
                    <Switch isSelected={autoMergeDuplicates} onChange={setAutoMergeDuplicates}>
                        {t('annotator.tools.ssim.options.autoMergeDuplicates')}
                    </Switch>
                    <Button variant={'accent'} onPress={accept} isDisabled={!hasPreview}>
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
