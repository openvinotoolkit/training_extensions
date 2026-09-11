// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Fragment } from 'react';

import type { PipelineComponentsHealth } from '@/api/types';
import { useTranslation } from '@/i18n';
import {
    ActionButton,
    Content,
    Dialog,
    DialogTrigger,
    dimensionValue,
    Flex,
    Grid,
    StatusLight,
    Text,
} from '@geti-ui/ui';
import { InfoOutline } from '@geti-ui/ui/icons';
import { usePipelineHealth } from 'hooks/api/pipeline.hook';

import { getComponentStatusMeta, getOverallStatusMeta, shouldShowPipelineHealthDetails } from './utils';

const COMPONENT_ORDER = ['source', 'sink', 'model'] as const;

type PipelineComponentsDetailsInfoProps = {
    components: PipelineComponentsHealth;
};

const PipelineComponentsDetailsInfo = ({ components }: PipelineComponentsDetailsInfoProps) => {
    const { t } = useTranslation();

    const componentLabels: Record<(typeof COMPONENT_ORDER)[number], string> = {
        source: t('inference.health.componentLabels.source'),
        sink: t('inference.health.componentLabels.sink'),
        model: t('inference.health.componentLabels.model'),
    };

    return (
        <DialogTrigger type={'popover'} placement={'top'}>
            <ActionButton isQuiet aria-label={'Pipeline component health'}>
                <InfoOutline />
            </ActionButton>
            <Dialog>
                <Content>
                    <Grid gap={'size-50'} columns={['max-content', 'max-content', 'auto']} alignContent={'start'}>
                        {COMPONENT_ORDER.map((key) => {
                            const { label, variant, message } = getComponentStatusMeta(components[key], t);

                            return (
                                <Fragment key={key}>
                                    <Text>{componentLabels[key]}</Text>
                                    <StatusLight
                                        variant={variant}
                                        UNSAFE_style={{ padding: 0, paddingRight: dimensionValue('size-50') }}
                                    >
                                        {label}
                                    </StatusLight>
                                    <Text marginStart={'size-150'}>{message}</Text>
                                </Fragment>
                            );
                        })}
                    </Grid>
                </Content>
            </Dialog>
        </DialogTrigger>
    );
};

export const PipelineHealth = () => {
    const { t } = useTranslation();
    const { data, isPending, isError } = usePipelineHealth();

    if (isPending || isError) {
        return null;
    }

    const { label, variant } = getOverallStatusMeta(data.status, t);
    const components = data.components;
    const showPipelineHealthDetails = components != null && shouldShowPipelineHealthDetails(components);

    return (
        <Flex alignItems={'center'} gap={'size-100'} height={'100%'}>
            <StatusLight role='status' variant={variant} UNSAFE_style={{ padding: 0, alignItems: 'center' }}>
                {label}
            </StatusLight>

            {showPipelineHealthDetails && <PipelineComponentsDetailsInfo components={components} />}
        </Flex>
    );
};
