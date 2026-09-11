// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { MqttSinkConfig } from '@/api/types';
import { useTranslation } from '@/i18n';
import { Flex, NumberField, Switch, TextField } from '@geti-ui/ui';

import { OutputFormats } from '../output-formats/output-formats.component';
import { RateLimitFields } from '../rate-limit/rate-limit-fields.component';

type MqttProps = {
    defaultState: MqttSinkConfig;
};

export const Mqtt = ({ defaultState }: MqttProps) => {
    const { t } = useTranslation();

    return (
        <Flex direction='column' gap='size-200'>
            <TextField isHidden label='id' name='id' defaultValue={defaultState.id} />
            <TextField
                width='100%'
                label={t('inference.sinks.fields.name')}
                name='name'
                defaultValue={defaultState.name || t('inference.sinks.defaultNames.mqtt')}
            />
            <TextField
                isRequired
                width='100%'
                label={t('inference.sinks.fields.brokerHost')}
                name='broker_host'
                defaultValue={defaultState.broker_host}
            />
            <Flex gap='size-200'>
                <TextField
                    flex='1'
                    label={t('inference.sinks.fields.topic')}
                    name='topic'
                    defaultValue={defaultState.topic}
                />
                <NumberField
                    label={t('inference.sinks.fields.brokerPort')}
                    name='broker_port'
                    minValue={0}
                    step={1}
                    defaultValue={defaultState.broker_port}
                />
            </Flex>
            <Flex gap='size-200' justifyContent='space-between' alignItems={'center'}>
                <RateLimitFields rateLimit={defaultState.rate_limit} />
            </Flex>

            <Flex>
                <Switch
                    name='auth_required'
                    alignSelf='end'
                    aria-label='Require Authentication'
                    defaultSelected={defaultState.auth_required}
                    key={defaultState.auth_required ? 'true' : 'false'}
                >
                    {t('inference.sinks.fields.authRequired')}
                </Switch>
            </Flex>

            <OutputFormats config={defaultState.output_formats} />
        </Flex>
    );
};
