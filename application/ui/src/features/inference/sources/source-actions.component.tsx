// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useMemo, useState } from 'react';

import type { SourceConfig } from '@/api/types';
import { useTranslation } from '@/i18n';
import { ActionButton, Flex, Loading, Text } from '@geti-ui/ui';
import { Back } from '@geti-ui/ui/icons';
import { usePipeline } from 'hooks/api/pipeline.hook';
import { isEmpty, orderBy } from 'lodash-es';

import { useSourcesQuery } from './api/use-sources';
import { EditSourceForm } from './edit-source-form.component';
import { SourcesList } from './source-list/source-list.component';
import { SourceOptions } from './source-options';

export const SourceActions = () => {
    const { t } = useTranslation();
    const [view, setView] = useState<'list' | 'options' | 'edit'>('list');
    const [currentSource, setCurrentSource] = useState<SourceConfig | null>(null);
    const { data: sources = [], isPending } = useSourcesQuery();
    const filteredSources = sources.filter((source) => source.source_type !== 'disconnected');
    const existingNames = useMemo(() => filteredSources.map((source) => source.name), [filteredSources]);

    const pipeline = usePipeline();
    const connectedSourceId = pipeline.data.source?.id;

    const handleShowList = () => {
        setView('list');
    };

    const handleAddSource = () => {
        setView('options');
    };

    const handleEditSource = (source: SourceConfig) => {
        setView('edit');
        setCurrentSource(source);
    };

    if (isPending) {
        return <Loading mode={'inline'} size='M' />;
    }

    if (view === 'edit' && !isEmpty(currentSource)) {
        return (
            <EditSourceForm
                config={currentSource}
                onSaved={handleShowList}
                onBackToList={handleShowList}
                connectedSourceId={connectedSourceId}
            />
        );
    }

    if (view === 'list') {
        const sourcesWithConnectedFirst = orderBy(filteredSources, (source) => source.id === connectedSourceId, 'desc');

        return (
            <SourcesList
                sources={sourcesWithConnectedFirst}
                onAddSource={handleAddSource}
                onEditSource={handleEditSource}
            />
        );
    }

    return (
        <SourceOptions onSaved={handleShowList} hasHeader={filteredSources.length > 0} existingNames={existingNames}>
            <Flex gap={'size-100'} marginBottom={'size-100'} alignItems={'center'}>
                <ActionButton isQuiet onPress={handleShowList}>
                    <Back />
                </ActionButton>

                <Text>{t('inference.sources.add.title')}</Text>
            </Flex>
        </SourceOptions>
    );
};
