// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import type { SinkConfig } from '@/api/types';
import { useTranslation } from '@/i18n';
import { ActionButton, Flex, Loading, Text } from '@geti-ui/ui';
import { Back } from '@geti-ui/ui/icons';
import { usePipeline } from 'hooks/api/pipeline.hook';
import { isEmpty, orderBy } from 'lodash-es';

import { useSinksQuery } from './api/use-sinks-query';
import { EditSinkForm } from './edit-sink-form.component';
import { SinkList } from './sink-list/sink-list.component';
import { SinkOptions } from './sink-options';

export const SinkActions = () => {
    const { t } = useTranslation();
    const [view, setView] = useState<'list' | 'options' | 'edit'>('list');
    const [currentSink, setCurrentSink] = useState<SinkConfig | null>(null);
    const { data: sinks = [], isPending } = useSinksQuery();
    const filteredSinks = sinks.filter((sink) => sink.sink_type !== 'disconnected');

    const pipeline = usePipeline();
    const connectedSinkId = pipeline.data.sink?.id;

    const handleShowList = () => {
        setView('list');
    };

    const handleAddSinks = () => {
        setView('options');
    };

    const handleEditSink = (sink: SinkConfig) => {
        setView('edit');
        setCurrentSink(sink);
    };

    if (isPending) {
        return <Loading mode={'inline'} size='M' />;
    }

    if (view === 'edit' && !isEmpty(currentSink)) {
        return (
            <EditSinkForm
                config={currentSink}
                onSaved={handleShowList}
                onBackToList={handleShowList}
                connectedSinkId={connectedSinkId}
            />
        );
    }

    if (view === 'list') {
        const sinksWithConnectedFirst = orderBy(filteredSinks, (sink) => sink.id === connectedSinkId, 'desc');

        return <SinkList sinks={sinksWithConnectedFirst} onAddSink={handleAddSinks} onEditSink={handleEditSink} />;
    }

    return (
        <SinkOptions onSaved={handleShowList} hasHeader>
            <Flex gap={'size-100'} marginBottom={'size-100'} alignItems={'center'}>
                <ActionButton isQuiet onPress={handleShowList}>
                    <Back />
                </ActionButton>

                <Text>{t('inference.sinks.add.title')}</Text>
            </Flex>
        </SinkOptions>
    );
};
