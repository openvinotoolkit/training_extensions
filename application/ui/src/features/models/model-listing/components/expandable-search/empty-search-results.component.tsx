// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { Flex, Heading } from '@geti-ui/ui';
import { Search } from '@geti-ui/ui/icons';

export const EmptySearchResults = () => {
    const { t } = useTranslation();

    return (
        <Flex direction={'column'} alignItems={'center'} justifyContent={'center'} gap={'size-200'} height={'100%'}>
            <Search />
            <Heading level={3}>{t('models.list.noModelsFound')}</Heading>
        </Flex>
    );
};
