// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ComponentProps, useState } from 'react';

import { useTranslation } from '@/i18n';
import { Checkbox, Flex, Item, ListView, Selection, Text } from '@geti-ui/ui';

import { isNonEmptyString } from '../../shared/util';

import classes from './multi-select-list.module.scss';

type ListViewProps = ComponentProps<typeof ListView>;

interface MultiSelectListProps<T extends string = string> extends Omit<
    ListViewProps,
    'selectionMode' | 'onSelectionChange' | 'items' | 'defaultSelectedKeys' | 'selectedKeys' | 'children'
> {
    name: string;
    label?: string;
    ariaLabel?: string;
    selectAllLabel?: string;
    defaultSelectedKeys: Set<T>;
    onSelectionChange?: (selectedIds: T[]) => void;
    items: { id: T; name: string }[];
}

export const MultiSelectList = <T extends string = string>({
    name,
    label,
    ariaLabel,
    selectAllLabel,
    items,
    onSelectionChange,
    defaultSelectedKeys,
    ...listProps
}: MultiSelectListProps<T>) => {
    const { t } = useTranslation();
    const [selectedLabels, setSelectedLabels] = useState<Set<T>>(defaultSelectedKeys);

    const resolvedSelectAllLabel = selectAllLabel ?? t('dataset.multiSelect.selectAll');

    const allItemSelected = selectedLabels.size === items.length && items.length > 0;

    const handleSelectAllItems = (isSelected: boolean) => {
        const selectedItems = isSelected ? new Set(items.map(({ id }) => id)) : new Set<T>();
        setSelectedLabels(selectedItems);
        onSelectionChange?.(Array.from(selectedItems));
    };

    const handleSelectChange = (keys: Selection) => {
        const selection = keys === 'all' ? new Set(items.map(({ id }) => id)) : (keys as Set<T>);
        setSelectedLabels(selection);
        onSelectionChange?.(Array.from(selection));
    };

    return (
        <Flex gap='size-100' direction='column'>
            {isNonEmptyString(label) && <Text UNSAFE_className={classes.label}>{label}</Text>}

            <Checkbox aria-label='Select all items' onChange={handleSelectAllItems} isSelected={allItemSelected}>
                {resolvedSelectAllLabel}
            </Checkbox>

            <ListView
                {...listProps}
                items={items}
                aria-label={ariaLabel ?? 'Multi-select list'}
                selectionMode='multiple'
                onSelectionChange={handleSelectChange}
                selectedKeys={selectedLabels}
            >
                {(item) => <Item key={item.id}>{item.name}</Item>}
            </ListView>

            <>
                {Array.from(selectedLabels).map((labelId) => (
                    <input key={labelId} type='hidden' name={name} value={labelId} />
                ))}
            </>
        </Flex>
    );
};
