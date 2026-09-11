// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { ActionButton, Divider, Flex, Text, View } from '@geti-ui/ui';

import { ReactComponent as EmptyDatasetImage } from '../../../../assets/empty-dataset.svg';
import {
    ActiveFiltersList,
    useClearAllFilters,
    useHasActiveFilters,
} from '../../gallery/active-filters/active-filters.component';
import { AnnotatorMediaFiltering } from '../../gallery/toolbar/media-filtering/annotator-media-filtering.component';

const NoMediaItemsMessage = () => {
    const { t } = useTranslation();

    return (
        <View
            marginTop={'size-100'}
            backgroundColor={'gray-100'}
            padding={'size-50'}
            borderColor={'transparent'}
            borderRadius={'medium'}
        >
            <Flex direction={'column'} alignItems={'center'} gap={'size-50'}>
                <View width={'size-1250'} height={'size-1250'}>
                    <EmptyDatasetImage height={'100%'} width={'100%'} />
                </View>
                <Text UNSAFE_style={{ textAlign: 'center' }}>
                    {t('dataset.empty.noMatches')} {t('dataset.empty.changeFilter')}
                </Text>
            </Flex>
        </View>
    );
};

type SidebarMediaFilterProps = {
    hasMediaItems: boolean;
};

export const SidebarMediaFilter = ({ hasMediaItems }: SidebarMediaFilterProps) => {
    const { t } = useTranslation();
    const hasActiveFilters = useHasActiveFilters();
    const handleClearAll = useClearAllFilters();

    return (
        <Flex direction={'column'} gap={'size-100'}>
            <Flex justifyContent={'space-between'} alignItems={'center'}>
                <AnnotatorMediaFiltering />
                {hasActiveFilters && (
                    <ActionButton isQuiet onPress={handleClearAll}>
                        {t('dataset.filtersActive.clearAll')}
                    </ActionButton>
                )}
            </Flex>
            <Divider size='S' orientation='horizontal' />
            {hasActiveFilters && (
                <Flex gap='size-100' wrap>
                    <ActiveFiltersList />
                </Flex>
            )}

            {hasActiveFilters && !hasMediaItems && <NoMediaItemsMessage />}
        </Flex>
    );
};
