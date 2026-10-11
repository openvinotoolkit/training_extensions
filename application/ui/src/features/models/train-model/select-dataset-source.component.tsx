// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useMemo } from 'react';

import { useTranslation } from '@/i18n';
import { Content, ContextualHelp, Heading, Item, Picker, Section } from '@geti-ui/ui';

import { DatasetSource, useTrainModelState } from './train-model-provider.component';

// The current dataset is a single self-describing entry, so it is grouped without a heading.
const SECTION_TITLE_KEYS = {
    current: undefined,
    view: 'models.training.setup.selectDataset.sections.views',
    revision: 'models.training.setup.selectDataset.sections.revisions',
} as const satisfies Record<DatasetSource['kind'], string | undefined>;

export const SelectDatasetSource = () => {
    const { t } = useTranslation();
    const { datasetSources, selectedDatasetSourceId, onSelectDatasetSourceId } = useTrainModelState();

    const sections = useMemo(
        () =>
            (['current', 'view', 'revision'] as const)
                .map((kind) => {
                    const titleKey = SECTION_TITLE_KEYS[kind];

                    return {
                        id: kind,
                        name: titleKey === undefined ? undefined : t(titleKey),
                        children: datasetSources.filter((source) => source.kind === kind),
                    };
                })
                .filter(({ children }) => children.length > 0),
        [t, datasetSources]
    );

    return (
        <Picker
            flex={1}
            items={sections}
            label={t('models.training.setup.selectDataset.label')}
            data-testid={'select-dataset'}
            selectedKey={selectedDatasetSourceId}
            onSelectionChange={(key) => onSelectDatasetSourceId(String(key))}
            contextualHelp={
                <ContextualHelp variant={'info'} placement={'top'}>
                    <Heading>{t('models.training.setup.selectDataset.helpTitle')}</Heading>
                    <Content>{t('models.training.setup.selectDataset.helpDescription')}</Content>
                </ContextualHelp>
            }
        >
            {(section) => (
                <Section key={section.id} title={section.name} items={section.children}>
                    {(item) => <Item key={item.id}>{item.name}</Item>}
                </Section>
            )}
        </Picker>
    );
};
