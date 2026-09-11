// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { Content, ContextualHelp, Heading, Item, Picker } from '@geti-ui/ui';

import { useTrainModelState } from './train-model-provider.component';

export const SelectModelRevision = () => {
    const { t } = useTranslation();
    const { modelRevisions, selectedModelRevisionId, onSelectModelRevisionId } = useTrainModelState();

    return (
        <Picker
            flex={1}
            items={modelRevisions}
            label={t('models.training.setup.selectModel.label')}
            selectedKey={selectedModelRevisionId}
            onSelectionChange={(key) => onSelectModelRevisionId(String(key))}
            contextualHelp={
                <ContextualHelp variant={'info'} placement={'top'}>
                    <Heading>{t('models.training.setup.selectModel.helpTitle')}</Heading>
                    <Content>{t('models.training.setup.selectModel.helpDescription')}</Content>
                </ContextualHelp>
            }
        >
            {(item) => <Item key={item.id}>{item.name}</Item>}
        </Picker>
    );
};
