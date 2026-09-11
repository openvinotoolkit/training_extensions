// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { Flex, Text } from '@geti-ui/ui';

type SelectedMediaCountProps = {
    count: number;
};

export const SelectedMediaCount = ({ count }: SelectedMediaCountProps) => {
    const { t } = useTranslation();

    return (
        <Flex direction={'column'} gap={'size-100'}>
            <Text>{t('dataset.views.selectedMediaCount', { count })}</Text>
        </Flex>
    );
};
