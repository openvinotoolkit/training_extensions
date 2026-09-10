// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Suspense } from 'react';

import { useTranslation } from '@/i18n';
import {
    ActionButton,
    Button,
    ButtonGroup,
    Content,
    Dialog,
    DialogTrigger,
    Divider,
    Heading,
    Loading,
    Tooltip,
    TooltipTrigger,
} from '@geti-ui/ui';
import { GraphChart } from '@geti-ui/ui/icons';

import { DatasetStatisticsContent } from './dataset-statistics-content.component';

export const DatasetStatistics = () => {
    const { t } = useTranslation();

    return (
        <DialogTrigger isDismissable>
            <TooltipTrigger>
                <ActionButton isQuiet aria-label={'dataset statistics'}>
                    <GraphChart />
                </ActionButton>
                <Tooltip>{t('dataset.statistics.label')}</Tooltip>
            </TooltipTrigger>
            {(close) => (
                <Dialog width={{ base: '90vw', L: '70vw' }}>
                    <Heading>{t('dataset.statistics.title')}</Heading>
                    <Divider />
                    <Content>
                        <Suspense fallback={<Loading size='M' />}>
                            <DatasetStatisticsContent />
                        </Suspense>
                    </Content>
                    <ButtonGroup>
                        <Button variant='secondary' onPress={close}>
                            {t('dataset.statistics.close')}
                        </Button>
                    </ButtonGroup>
                </Dialog>
            )}
        </DialogTrigger>
    );
};
