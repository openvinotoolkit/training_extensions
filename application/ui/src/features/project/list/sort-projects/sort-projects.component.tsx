// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Item, Picker, Section } from '@geti-ui/ui';
import { useTranslation } from 'react-i18next';

import { SORT_BY_OPTIONS, SortBy } from './utils';

import classes from './sort-projects.module.scss';

type SortProjectsProps = {
    sortBy: SortBy;
    onSort: (sortBy: SortBy) => void;
};

export { SORT_BY_HANDLERS } from './utils';

export const SortProjects = ({ sortBy, onSort }: SortProjectsProps) => {
    const { t } = useTranslation();

    return (
        <Picker
            isQuiet
            items={SORT_BY_OPTIONS}
            selectedKey={sortBy}
            onSelectionChange={(key) => onSort(key as SortBy)}
            labelAlign={'start'}
            labelPosition={'side'}
            label={t('project.list.sort.label')}
            UNSAFE_className={classes.sortProjects}
        >
            {(item) => {
                return (
                    <Section key={item.map((option) => option.key).join('-')}>
                        {item.map((option) => (
                            <Item key={option.key} textValue={t(option.nameKey)}>
                                {t(option.nameKey)}
                            </Item>
                        ))}
                    </Section>
                );
            }}
        </Picker>
    );
};
