// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { Divider, Flex, Text } from '@geti-ui/ui';
import { useDatasetMediaWithReviewStatus } from 'hooks/use-dataset-media-with-review-status.hook';

type TotalItemsProps = {
    totalSelectedElements: number;
};

export const TotalItems = ({ totalSelectedElements }: TotalItemsProps) => {
    const { t } = useTranslation();
    const { totalCount } = useDatasetMediaWithReviewStatus();

    if (totalCount === 0) {
        return null;
    }

    const hasSelectedElements = totalSelectedElements > 0;

    return (
        <Flex gap={'size-100'}>
            {hasSelectedElements && (
                <>
                    <Text>{t('dataset.gallery.selectedCount', { count: totalSelectedElements })}</Text>
                    <Divider orientation={'vertical'} size={'S'} />
                </>
            )}

            <Text>{t('dataset.gallery.totalMediaCount', { count: totalCount })}</Text>
        </Flex>
    );
};
