// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { SourceConfig } from '@/api/types';
import { ConnectionStatusBadge } from '@/components/connection-status-badge/connection-status-badge.component';
import { Button, dimensionValue, Flex, Text } from '@geti-ui/ui';
import { Add as AddIcon } from '@geti-ui/ui/icons';
import { clsx } from 'clsx';
import { usePipeline } from 'hooks/api/pipeline.hook';
import { isEqual } from 'lodash-es';

import { getErrorMessage } from '../../../../query-client/query-client';
import { removeUnderscore } from '../../util';
import { useTestSource } from '../api/use-test-source';
import { SourceMenu } from '../source-menu/source-menu.component';
import { SettingsList } from './settings-list/settings-list.component';
import { SourceIcon } from './source-icon/source-icon.component';

import classes from './source-list.module.scss';

type SourcesListProps = {
    sources: SourceConfig[];
    onAddSource: () => void;
    onEditSource: (config: SourceConfig) => void;
};

type SourceListItemProps = {
    source: SourceConfig;
    isConnected: boolean;
    isPipelineRunning: boolean;
    onEditSource: (config: SourceConfig) => void;
};

const SourceListItem = ({ source, isConnected, onEditSource, isPipelineRunning }: SourceListItemProps) => {
    const { data, error, isError, isFetched, isFetching, refetch } = useTestSource(String(source.id));
    const showConnectionStatusBadge = isConnected || isFetched || isFetching || isError;

    const handleTestConnection = async () => {
        void refetch();
    };

    return (
        <Flex
            key={source.id}
            gap='size-200'
            direction='column'
            data-testid={`source-card-${source.name}`}
            UNSAFE_className={clsx(classes.card, { [classes.activeCard]: isConnected })}
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
                <SourceIcon type={source.source_type} />

                <Flex direction={'column'} gap={'size-100'}>
                    <Text UNSAFE_className={classes.title}>{source.name}</Text>
                    <Flex gap={'size-100'} alignItems={'center'}>
                        <Text UNSAFE_className={classes.type}>{removeUnderscore(source.source_type)}</Text>
                    </Flex>
                </Flex>
            </Flex>

            <Flex justifyContent={'space-between'}>
                <SettingsList source={source} />

                <SourceMenu
                    id={String(source.id)}
                    name={source.name}
                    isConnected={isConnected}
                    onEdit={() => onEditSource(source)}
                    isPipelineRunning={isPipelineRunning}
                    onTest={handleTestConnection}
                />
            </Flex>
        </Flex>
    );
};

export const SourcesList = ({ sources, onAddSource, onEditSource }: SourcesListProps) => {
    const pipeline = usePipeline();
    const currentSourceId = pipeline.data.source?.id;
    const isPipelineRunning = pipeline.data.status === 'running';

    return (
        <Flex
            gap={'size-200'}
            direction={'column'}
            UNSAFE_style={{ overflow: 'auto', padding: dimensionValue('size-10') }}
        >
            <Button variant='secondary' height={'size-800'} UNSAFE_className={classes.addSource} onPress={onAddSource}>
                <AddIcon /> Add new source
            </Button>

            {sources.map((source) => (
                <SourceListItem
                    key={source.id}
                    source={source}
                    isConnected={isEqual(currentSourceId, source.id)}
                    onEditSource={onEditSource}
                    isPipelineRunning={isPipelineRunning}
                />
            ))}
        </Flex>
    );
};
