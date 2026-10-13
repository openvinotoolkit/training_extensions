// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Model } from '@/api/types';
import { OnnxLogo, OpenVinoLogo, PyTorchLogo } from '@/assets/icons';
import { useTranslation } from '@/i18n';
import { Flex, Item, TabList, TabPanels, Tabs, Text } from '@geti-ui/ui';
import { isEmpty } from 'lodash-es';

import { ModelVariantTable } from './model-variant-table.component';
import { QuantizationRow } from './quantization-row.component';

import classes from './model-variant-tabs.module.scss';

type ModelVariantsTabsProps = {
    model: Model;
};

const isQuantizationDisabled = (model: Model) => {
    return Boolean(
        model.files_deleted ||
        model.variants?.some(
            (variant) => variant.format === 'openvino' && variant.precision === 'fp16' && variant.files_deleted
        )
    );
};

export const ModelVariantsTabs = ({ model }: ModelVariantsTabsProps) => {
    const { t } = useTranslation();

    if (isEmpty(model.variants)) {
        return (
            <Flex justifyContent={'center'} alignItems={'center'} height={'size-3000'}>
                <Text>{t('models.variants.empty')}</Text>
            </Flex>
        );
    }

    return (
        <Tabs aria-label='Model variants' UNSAFE_className={classes.tabs}>
            <TabList>
                <Item aria-label='openvino tab' key='openvino' textValue='openvino'>
                    <OpenVinoLogo />
                </Item>
                <Item aria-label='pytorch tab' key='pytorch' textValue='pytorch'>
                    <PyTorchLogo />
                </Item>
                <Item aria-label='onnx tab' key='onnx' textValue='onnx'>
                    <OnnxLogo />
                </Item>
            </TabList>
            <TabPanels width={0} minWidth={'100%'} UNSAFE_className={classes.tabPanels}>
                <Item key='openvino'>
                    <ModelVariantTable model={model} format='openvino' />
                    <QuantizationRow model={model} isDisabled={isQuantizationDisabled(model)} />
                </Item>
                <Item key='pytorch'>
                    <ModelVariantTable model={model} format='pytorch' />
                </Item>
                <Item key='onnx'>
                    <ModelVariantTable model={model} format='onnx' />
                </Item>
            </TabPanels>
        </Tabs>
    );
};
