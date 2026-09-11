// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { SinkConfig } from '@/api/types';
import { ConnectionStatusBadge } from '@/components/connection-status-badge/connection-status-badge.component';
import { useTranslation } from '@/i18n';
import { Button, dimensionValue, Flex, Text } from '@geti-ui/ui';
import { Add as AddIcon } from '@geti-ui/ui/icons';
import { clsx } from 'clsx';
import { isEqual } from 'lodash-es';

import { usePipeline } from '../../../../hooks/api/pipeline.hook';
import { getErrorMessage } from '../../../../query-client/query-client';
import { useTestSink } from '../api/use-test-sink';
import { SettingsList } from './settings-list/settings-list.component';
import { SinkIcon } from './sink-icon/sink-icon.component';
import { SinkMenu } from './sink-menu/sink-menu.component';

import classes from './sink-list.module.scss';

const SINK_TYPE_LABEL_KEYS = {
    folder: 'inference.sinks.types.folder',
    mqtt: 'inference.sinks.types.mqtt',
    webhook: 'inference.sinks.types.webhook',
    ros: 'inference.sinks.types.ros',
} as const satisfies Record<SinkConfig['sink_type'], string>;

type SinksListProps = {
    sinks: SinkConfig[];
    onAddSink: () => void;
    onEditSink: (config: SinkConfig) => void;
};

type SinksListItemProps = {
    sink: SinkConfig;
    isConnected: boolean;
    onEditSink: (config: SinkConfig) => void;
};

const SinkListItem = ({ sink, isConnected, onEditSink }: SinksListItemProps) => {
    const { t } = useTranslation();
    const { data, error, isError, isFetched, isFetching, refetch } = useTestSink(sink.id);
    const showConnectionStatusBadge = isConnected || isFetched || isFetching || isError;

    const handleTestConnection = async () => {
        void refetch();
    };

    return (
        <Flex
            key={sink.id}
            gap='size-200'
            direction='column'
            data-testid={`sink-card-${sink.name}`}
            UNSAFE_className={clsx(classes.card, {
                [classes.activeCard]: isConnected,
            })}
        >
            {showConnectionStatusBadge && (
                <ConnectionStatusBadge
                    isInUse={isConnected}
                    isUnreachable={isError || (isFetched && data?.reachable === false)}
                    isPending={isFetching}
                    errorMessage={isError ? getErrorMessage(error) : undefined}
                />
            )}
            <Flex alignItems={'center'} gap={'size-200'}>
                <SinkIcon type={sink.sink_type} />

                <Flex direction={'column'} gap={'size-100'}>
                    <Text UNSAFE_className={classes.title}>{sink.name}</Text>
                    <Flex gap={'size-100'} alignItems={'center'}>
                        <Text UNSAFE_className={classes.type}>{t(SINK_TYPE_LABEL_KEYS[sink.sink_type])}</Text>
                    </Flex>
                </Flex>
            </Flex>

            <Flex justifyContent={'space-between'}>
                <SettingsList sink={sink} />

                <SinkMenu
                    id={sink.id}
                    name={sink.name}
                    isConnected={isConnected}
                    onEdit={() => onEditSink(sink)}
                    onTest={handleTestConnection}
                />
            </Flex>
        </Flex>
    );
};

export const SinkList = ({ sinks, onAddSink, onEditSink }: SinksListProps) => {
    const { t } = useTranslation();
    const pipeline = usePipeline();
    const currentSinkId = pipeline.data.sink?.id;

    return (
        <Flex
            gap={'size-200'}
            direction={'column'}
            UNSAFE_style={{ overflow: 'auto', padding: dimensionValue('size-10') }}
        >
            <Button variant='secondary' height={'size-800'} UNSAFE_className={classes.addSink} onPress={onAddSink}>
                <AddIcon /> {t('inference.sinks.list.addSink')}
            </Button>

            {sinks.map((sink) => (
                <SinkListItem
                    key={sink.id}
                    sink={sink}
                    isConnected={isEqual(currentSinkId, sink.id)}
                    onEditSink={onEditSink}
                />
            ))}
        </Flex>
    );
};
