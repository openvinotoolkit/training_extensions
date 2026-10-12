// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { Item, Picker } from '@geti-ui/ui';
import type { SortDirection } from 'hooks/sort-direction.interface';
import { useDatasetFiltersSearchParams } from 'hooks/use-dataset-filters-search-params.hook';

import classes from './sort-media.module.scss';

const SORT_OPTIONS = [
    { key: 'desc', nameKey: 'dataset.gallery.sort.uploadDateDescending' },
    { key: 'asc', nameKey: 'dataset.gallery.sort.uploadDateAscending' },
] as const;

export const SortMedia = () => {
    const { i18n, t } = useTranslation();
    const { sortDirection, setSortDirection } = useDatasetFiltersSearchParams();

    return (
        <Picker
            key={i18n.resolvedLanguage ?? i18n.language}
            isQuiet
            selectedKey={sortDirection}
            onSelectionChange={(key) => setSortDirection(key as SortDirection)}
            labelAlign={'start'}
            labelPosition={'side'}
            label={t('dataset.gallery.sort.label')}
            UNSAFE_className={classes.sortMedia}
        >
            {SORT_OPTIONS.map((option) => {
                const name = t(option.nameKey);

                return (
                    <Item key={option.key} textValue={name}>
                        {name}
                    </Item>
                );
            })}
        </Picker>
    );
};
