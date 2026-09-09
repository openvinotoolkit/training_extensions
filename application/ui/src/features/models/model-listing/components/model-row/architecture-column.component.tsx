// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { ModelArchitectureWithPerformanceCategory } from '@/api/types';
import { Flex, Text } from '@geti-ui/ui';

import { PerformanceCategoryBadge } from './performance-category-badge.component';

import classes from './model-row.module.scss';

type ArchitectureColumnProps = {
    architectureId: string;
    architecture: ModelArchitectureWithPerformanceCategory | undefined;
};

export const ArchitectureColumn = ({ architectureId, architecture }: ArchitectureColumnProps) => {
    // For TIMM, and as a general fallback, we display the architecture ID (and no license)
    if (architecture === undefined) {
        return <Text UNSAFE_className={classes.smallText}>{architectureId}</Text>;
    }

    return (
        <Flex direction={'column'} gap={'size-100'}>
            <Text UNSAFE_className={classes.smallText}>
                {architecture.name} ({architecture.license})
            </Text>
            {architecture.performanceCategory !== undefined && (
                <PerformanceCategoryBadge
                    id={'architecture-name'}
                    performanceCategory={architecture.performanceCategory}
                />
            )}
        </Flex>
    );
};
