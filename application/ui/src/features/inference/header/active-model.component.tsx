// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useMemo } from 'react';

import { useTranslation } from '@/i18n';
import { Item, Key, Picker } from '@geti-ui/ui';
import { usePatchPipeline } from 'hooks/api/pipeline.hook';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { isEmpty } from 'lodash-es';

import { useGetActiveModel } from '../../models/hooks/api/use-get-active-model.hook';
import { useGetSuccessfulModels } from '../../models/hooks/api/use-get-models.hook';
import { getAllModelsWithOpenVINOVariants, getModelIdentifierPayload } from '../../models/utils';

export const ActiveModel = () => {
    const { t } = useTranslation();
    const { data: models } = useGetSuccessfulModels();
    const activeModel = useGetActiveModel();
    const projectId = useProjectIdentifier();
    const updatePipeline = usePatchPipeline();

    const allModelsWithOpenVinoQuantizedModels = useMemo(() => getAllModelsWithOpenVINOVariants(models), [models]);

    const handleChange = (key: Key | null) => {
        if (key === null) {
            return;
        }

        const selectedModel = allModelsWithOpenVinoQuantizedModels.find((model) => model.modelVariantId === key);

        if (selectedModel === undefined) {
            return;
        }

        const body = getModelIdentifierPayload(selectedModel);

        updatePipeline.mutate({
            params: { path: { project_id: projectId } },
            body,
        });
    };

    if (isEmpty(allModelsWithOpenVinoQuantizedModels)) {
        return null;
    }

    return (
        <>
            <Picker
                aria-label={'active model'}
                label={t('inference.pipeline.activeModel.label')}
                labelPosition={'side'}
                items={allModelsWithOpenVinoQuantizedModels}
                onSelectionChange={handleChange}
                selectedKey={activeModel.model_variant_id ?? null}
                minWidth={'size-3400'}
            >
                {(item) => <Item key={item.modelVariantId}>{item.name}</Item>}
            </Picker>
        </>
    );
};
