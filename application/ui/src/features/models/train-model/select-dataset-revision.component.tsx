// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { Content, ContextualHelp, Heading, Item, Picker } from '@geti-ui/ui';

import { useTrainModelState } from './train-model-provider.component';

export const SelectDatasetRevision = () => {
    const { t } = useTranslation();
    const { datasetRevisions, selectedDatasetRevisionId, onSelectDatasetRevisionId } = useTrainModelState();

    return (
        <>
            <Picker
                flex={1}
                items={datasetRevisions}
                label={t('models.training.setup.selectDataset.label')}
                selectedKey={selectedDatasetRevisionId}
                onSelectionChange={(key) => onSelectDatasetRevisionId(String(key))}
                contextualHelp={
                    <ContextualHelp variant={'info'} placement={'top'}>
                        <Heading>{t('models.training.setup.selectDataset.helpTitle')}</Heading>
                        <Content>{t('models.training.setup.selectDataset.helpDescription')}</Content>
                    </ContextualHelp>
                }
            >
                {(item) => <Item key={item.id}>{item.name}</Item>}
            </Picker>
        </>
    );
};
