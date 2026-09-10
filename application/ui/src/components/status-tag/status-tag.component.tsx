// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { Flex, Text } from '@geti-ui/ui';
import { clsx } from 'clsx';

import classes from './status-tag.module.scss';

interface StatusTagProps {
    isError?: boolean;
    isConnected?: boolean;
}

export const StatusTag = ({ isConnected = false, isError = false }: StatusTagProps) => {
    const { t } = useTranslation();

    if (isError) {
        return (
            <Flex gap={'size-75'} alignItems={'center'} UNSAFE_className={classes.container}>
                <div className={classes.status}></div>
                <Text>{t('dataset.jobs.status.error')}</Text>
            </Flex>
        );
    }
    return (
        <Flex gap={'size-75'} alignItems={'center'} UNSAFE_className={classes.container}>
            <div
                className={clsx({
                    [classes.status]: true,
                    [classes.connected]: isConnected,
                    [classes.disconnected]: !isConnected,
                })}
            ></div>
            <Text>{isConnected ? t('dataset.jobs.status.connected') : t('dataset.jobs.status.disconnected')}</Text>
        </Flex>
    );
};
