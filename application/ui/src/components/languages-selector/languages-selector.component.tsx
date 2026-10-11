// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Earth } from '@/assets/icons';
import { SUPPORTED_LANGUAGES, useTranslation } from '@/i18n';
import { ActionButton, Icon, Item, Key, Menu, MenuTrigger, Text } from '@geti-ui/ui';
import { capitalize } from 'lodash-es';

import classes from './languages-selector.module.scss';

const getLanguageName = (language: string): string =>
    new Intl.DisplayNames([language], { type: 'language' }).of(language) ?? language;

const languages = SUPPORTED_LANGUAGES.map((language) => ({
    id: language,
    name: getLanguageName(language),
}));

export const LanguagesSelector = () => {
    const { i18n } = useTranslation();
    const current = i18n.resolvedLanguage ?? i18n.language;

    const handleChange = (key: Key) => {
        void i18n.changeLanguage(String(key));
    };

    return (
        <MenuTrigger>
            <ActionButton isQuiet aria-label='Language'>
                <Icon UNSAFE_className={classes.icon}>
                    <Earth />
                </Icon>
                <Text>{current.toLocaleUpperCase()}</Text>
            </ActionButton>
            <Menu items={languages} onAction={handleChange}>
                {(item) => <Item key={item.id}>{capitalize(item.name)}</Item>}
            </Menu>
        </MenuTrigger>
    );
};
