// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Model } from '@/api/types';
import { useTranslation } from '@/i18n';
import { Button, Content, ContextualHelp, DialogTrigger, Flex, Text } from '@geti-ui/ui';

import { QuantizationDialog } from './quantization-dialog/quantization-dialog.component';

type QuantizationRowProps = {
    model: Model;
    isDisabled?: boolean;
};
export const QuantizationRow = ({ model, isDisabled = false }: QuantizationRowProps) => {
    const { t } = useTranslation();

    return (
        <Flex marginTop={'size-150'} alignItems={'center'} justifyContent={'space-between'}>
            <Flex>
                <Text>{t('models.optimize.description')}</Text>
                <ContextualHelp>
                    <Content>{t('models.optimize.contextualHelp')}</Content>
                </ContextualHelp>
            </Flex>
            <DialogTrigger>
                <Button variant={'secondary'} isDisabled={isDisabled}>
                    {t('models.optimize.start')}
                </Button>
                {(close) => <QuantizationDialog model={model} onClose={close} />}
            </DialogTrigger>
        </Flex>
    );
};
