// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { Badge, Text } from '@geti-ui/ui';
import { clsx } from 'clsx';

import classes from './active-project-badge.module.scss';

type ActiveProjectBadgeProps = {
    size?: 'S' | 'M';
};

export const ActiveProjectBadge = ({ size = 'M' }: ActiveProjectBadgeProps) => {
    const { t } = useTranslation();

    return (
        <Badge variant={'neutral'} UNSAFE_className={clsx(classes.activeTag, { [classes.small]: size === 'S' })}>
            <Text>{t('project.list.activeBadge')}</Text>
        </Badge>
    );
};
