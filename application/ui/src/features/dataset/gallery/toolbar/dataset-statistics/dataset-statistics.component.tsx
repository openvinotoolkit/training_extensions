// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Suspense } from 'react';

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
    return (
        <DialogTrigger isDismissable>
            <TooltipTrigger>
                <ActionButton isQuiet aria-label={'dataset statistics'}>
                    <GraphChart />
                </ActionButton>
                <Tooltip>Dataset statistics</Tooltip>
            </TooltipTrigger>
            {(close) => (
                <Dialog width={{ base: '90vw', L: '70vw' }}>
                    <Heading>Dataset Statistics</Heading>
                    <Divider />
                    <Content>
                        <Suspense fallback={<Loading size='M' />}>
                            <DatasetStatisticsContent />
                        </Suspense>
                    </Content>
                    <ButtonGroup>
                        <Button variant='secondary' onPress={close}>
                            Close
                        </Button>
                    </ButtonGroup>
                </Dialog>
            )}
        </DialogTrigger>
    );
};
