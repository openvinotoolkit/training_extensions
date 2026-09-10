// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { useTranslation } from '@/i18n';
import { Flex, Loading, Text } from '@geti-ui/ui';
import { Checkmark, CloseSmall } from '@geti-ui/ui/icons';
import { clsx } from 'clsx';

import classes from './connection-status-badge.module.scss';

type ConnectionStatusBadgeProps = {
    isInUse: boolean;
    isUnreachable: boolean;
    isPending: boolean;
    errorMessage?: string;
};

type BadgeState = {
    icon: ReactNode;
    label: string;
    isInUseContainer?: boolean;
};

export const ConnectionStatusBadge = ({
    isInUse,
    isUnreachable,
    isPending,
    errorMessage,
}: ConnectionStatusBadgeProps) => {
    const { t } = useTranslation();

    const badgeState: BadgeState = isPending
        ? {
              icon: <Loading mode='inline' size='S' />,
              label: t('inference.connection.testing'),
          }
        : isInUse
          ? {
                icon: <Checkmark size='S' UNSAFE_className={classes.inUseIcon} />,
                label: t('inference.connection.inUse'),
                isInUseContainer: true,
            }
          : isUnreachable
            ? {
                  icon: <CloseSmall className={classes.unreachableIcon} />,
                  label: errorMessage ?? t('inference.connection.unreachable'),
              }
            : {
                  icon: <Checkmark size='S' UNSAFE_className={classes.readyIcon} />,
                  label: t('inference.connection.ready'),
              };

    return (
        <Flex
            gap={'size-75'}
            alignItems={'center'}
            UNSAFE_className={clsx(classes.container, { [classes.inUseContainer]: badgeState.isInUseContainer })}
        >
            {badgeState.icon}
            <Text>{badgeState.label}</Text>
        </Flex>
    );
};
