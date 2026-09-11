// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import {
    ActionButton,
    Content,
    Dialog,
    DialogTrigger,
    Divider,
    Flex,
    Heading,
    Text,
    Tooltip,
    TooltipTrigger,
} from '@geti-ui/ui';
import { Adjustments, Close } from '@geti-ui/ui/icons';

import { CanvasSettings } from './canvas-settings.component';

import styles from './settings.module.scss';

export const Settings = () => {
    const { t } = useTranslation();

    return (
        <DialogTrigger type={'popover'} hideArrow placement={'top'}>
            <TooltipTrigger>
                <ActionButton isQuiet aria-label={'Settings'}>
                    <Adjustments />
                </ActionButton>
                <Tooltip>{t('annotator.canvas.settingsTitle')}</Tooltip>
            </TooltipTrigger>
            {(close) => (
                <Dialog UNSAFE_className={styles.settingsDialog}>
                    <Heading>
                        <Flex justifyContent={'space-between'} alignItems={'center'}>
                            <Text>{t('annotator.canvas.settingsTitle')}</Text>
                            <ActionButton isQuiet onPress={close} aria-label={'Close settings'}>
                                <Close />
                            </ActionButton>
                        </Flex>
                    </Heading>
                    <Divider size={'S'} />
                    <Content>
                        <CanvasSettings />
                    </Content>
                </Dialog>
            )}
        </DialogTrigger>
    );
};
