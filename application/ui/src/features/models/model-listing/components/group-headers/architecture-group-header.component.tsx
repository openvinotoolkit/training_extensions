// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { ModelArchitectureWithPerformanceCategory } from '@/api/types';
import { useTranslation } from '@/i18n';
import { dimensionValue, Flex, Heading, Text } from '@geti-ui/ui';

import { PerformanceCategoryBadge } from '../model-row/performance-category-badge.component';

type ArchitectureGroupHeaderProps = {
    architecture: ModelArchitectureWithPerformanceCategory | undefined;
};

export const ArchitectureGroupHeader = ({ architecture }: ArchitectureGroupHeaderProps) => {
    const { t } = useTranslation();

    // Should never happen, but just in case
    if (architecture === undefined) {
        return <Text>{t('models.list.unknownArchitecture')}</Text>;
    }

    return (
        <Flex alignItems={'center'} gap={'size-200'} marginBottom={'size-225'}>
            <Heading level={2} UNSAFE_style={{ fontSize: dimensionValue('size-300') }}>
                {architecture.name}
            </Heading>

            {architecture.performanceCategory !== undefined && (
                <PerformanceCategoryBadge
                    id={'architecture-name'}
                    performanceCategory={architecture.performanceCategory}
                />
            )}
        </Flex>
    );
};
