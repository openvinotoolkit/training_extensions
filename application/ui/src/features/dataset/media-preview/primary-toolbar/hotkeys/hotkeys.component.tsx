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
import { Close, Hotkeys as HotkeysIcon } from '@geti-ui/ui/icons';

import { HotkeysList } from './hotkeys-list.component';

import styles from './hotkeys.module.scss';

export const Hotkeys = () => {
    const { t } = useTranslation();

    return (
        <DialogTrigger type={'popover'} hideArrow placement={'top'}>
            <TooltipTrigger>
                <ActionButton isQuiet aria-label={'Hotkeys'}>
                    <HotkeysIcon />
                </ActionButton>
                <Tooltip>{t('annotator.hotkeys.title')}</Tooltip>
            </TooltipTrigger>
            {(close) => (
                <Dialog UNSAFE_className={styles.hotkeysDialog}>
                    <Heading>
                        <Flex justifyContent={'space-between'} alignItems={'center'}>
                            <Text>{t('annotator.hotkeys.title')}</Text>
                            <ActionButton isQuiet onPress={close} aria-label={'Close hotkeys'}>
                                <Close />
                            </ActionButton>
                        </Flex>
                    </Heading>
                    <Divider size={'S'} />
                    <Content>
                        <HotkeysList />
                    </Content>
                </Dialog>
            )}
        </DialogTrigger>
    );
};
