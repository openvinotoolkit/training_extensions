// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { ModelArchitectureWithPerformanceCategory } from '@/api/types';
import { useTranslation } from '@/i18n';
import { dimensionValue, Flex, Heading, Text } from '@geti-ui/ui';

import { ModelLicenseLink } from '../../../components/model-license-link.component';
import { PerformanceCategoryBadge } from '../model-row/performance-category-badge.component';

type ArchitectureGroupHeaderProps = {
    architectureId: string;
    architecture: ModelArchitectureWithPerformanceCategory | undefined;
};

export const ArchitectureGroupHeader = ({ architectureId, architecture }: ArchitectureGroupHeaderProps) => {
    const { t } = useTranslation();

    // For TIMM backbones or removed manifests the architecture isn't in the catalog, so show the raw id
    if (architecture === undefined) {
        return (
            <Flex alignItems={'center'} gap={'size-200'} marginBottom={'size-225'}>
                <Heading level={2} UNSAFE_style={{ fontSize: dimensionValue('size-300') }}>
                    {architectureId}
                </Heading>
            </Flex>
        );
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

            <Text>
                {t('license.label')}
                <ModelLicenseLink license={architecture.license} />
            </Text>
        </Flex>
    );
};
