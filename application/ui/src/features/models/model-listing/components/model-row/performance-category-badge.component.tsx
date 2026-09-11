// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { Text } from '@geti-ui/ui';
import { capitalize } from 'lodash-es';

import { ReactComponent as ThumbsUp } from '../../../../../assets/icons/thumbs-up.svg';
import { ModelBadge } from './model-badge.component';

type PerformanceCategoryBadgeProps = {
    performanceCategory: string;
    id?: string;
    color?: string;
};

type PerformanceCategoryLabelKey =
    | 'models.performance.categories.balance'
    | 'models.performance.categories.speed'
    | 'models.performance.categories.accuracy';

const PERFORMANCE_CATEGORY_LABEL_KEYS: Record<string, PerformanceCategoryLabelKey> = {
    balance: 'models.performance.categories.balance',
    speed: 'models.performance.categories.speed',
    accuracy: 'models.performance.categories.accuracy',
};

export const PerformanceCategoryBadge = ({ performanceCategory, id, color }: PerformanceCategoryBadgeProps) => {
    const { t } = useTranslation();
    const labelKey = PERFORMANCE_CATEGORY_LABEL_KEYS[performanceCategory.toLowerCase()];

    return (
        <ModelBadge id={id} color={color}>
            <ThumbsUp />
            <Text>{labelKey ? t(labelKey) : capitalize(performanceCategory)}</Text>
        </ModelBadge>
    );
};
