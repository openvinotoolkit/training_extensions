// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { ActionButton, Item, Menu, MenuTrigger } from '@geti-ui/ui';
import { HelpIcon } from '@geti-ui/ui/icons';

import { openExternalUrl } from '../../platform/open-external-url';

export const HelpMenu = () => {
    const { t } = useTranslation();

    return (
        <MenuTrigger>
            <ActionButton isQuiet aria-label='Help'>
                <HelpIcon />
            </ActionButton>
            <Menu onAction={(url) => openExternalUrl(String(url))}>
                <Item key='https://docs.geti.intel.com/'>{t('navigation.help.documentation')}</Item>
                <Item key='https://github.com/open-edge-platform/geti/releases'>{t('navigation.help.whatsNew')}</Item>
                <Item key='https://github.com/open-edge-platform/geti/issues'>{t('navigation.help.reportIssue')}</Item>
            </Menu>
        </MenuTrigger>
    );
};
