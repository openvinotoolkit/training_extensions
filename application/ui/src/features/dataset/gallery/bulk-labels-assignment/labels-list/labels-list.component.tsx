// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useDeferredValue, useMemo, useState } from 'react';

import type { Label } from '@/api/types';
import { useTranslation } from '@/i18n';
import { Flex, Item, ListView, Selection, Text, TextField, View } from '@geti-ui/ui';
import { isEmpty } from 'lodash-es';

import { EMPTY_LABEL_ID } from '../../../../../shared/annotator/labels';

type LabelsListProps = {
    ariaLabel: string;
    labels: Label[];
    selectedLabels: Set<string>;
    onSelectedLabelsChange: (selectedLabels: Set<string>) => void;
    isMultiple: boolean;
};

const INITIAL_SEARCH_PHRASE = '';

export const LabelsList = ({
    labels,
    ariaLabel,
    selectedLabels,
    onSelectedLabelsChange,
    isMultiple,
}: LabelsListProps) => {
    const { t } = useTranslation();
    const [searchPhrase, setSearchPhrase] = useState<string>(INITIAL_SEARCH_PHRASE);
    const deferredSearchPhrase = useDeferredValue(searchPhrase, INITIAL_SEARCH_PHRASE);

    const filteredLabels = useMemo(() => {
        return labels.filter((label) => label.name.toLowerCase().includes(deferredSearchPhrase.toLowerCase()));
    }, [deferredSearchPhrase, labels]);

    const hasNoSearchResults = !isEmpty(deferredSearchPhrase) && isEmpty(filteredLabels);

    const handleSelectChange = (keys: Selection) => {
        const newKeys = new Set(keys);

        const hasEmptyLabelSelected =
            selectedLabels.has(EMPTY_LABEL_ID) && newKeys.has(EMPTY_LABEL_ID) && newKeys.size > 1;

        if (hasEmptyLabelSelected) {
            newKeys.delete(EMPTY_LABEL_ID);
            onSelectedLabelsChange(newKeys as Set<string>);
        } else if (!selectedLabels.has(EMPTY_LABEL_ID) && newKeys.has(EMPTY_LABEL_ID)) {
            onSelectedLabelsChange(new Set([EMPTY_LABEL_ID]));
        } else {
            onSelectedLabelsChange(newKeys as Set<string>);
        }
    };

    return (
        <Flex gap='size-200' direction='column' flex={1} minHeight={0}>
            <TextField
                aria-label={'Search labels'}
                value={searchPhrase}
                onChange={setSearchPhrase}
                placeholder={t('dataset.filters.searchLabels')}
            />

            {hasNoSearchResults ? (
                <Text>{t('dataset.bulkLabels.noResults')}</Text>
            ) : (
                <View flex={1} UNSAFE_style={{ overflowY: 'auto' }}>
                    <ListView
                        items={filteredLabels}
                        aria-label={ariaLabel}
                        selectionMode={isMultiple ? 'multiple' : 'single'}
                        onSelectionChange={handleSelectChange}
                        selectedKeys={selectedLabels}
                    >
                        {(item) => <Item key={item.id}>{item.name}</Item>}
                    </ListView>
                </View>
            )}
        </Flex>
    );
};
