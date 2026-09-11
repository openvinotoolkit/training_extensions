// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { SinkConfig, SinkOutputFormats, WebhookSinkConfig } from '@/api/types';
import { useTranslation } from '@/i18n';

import { formatRateLimit, OutputFormat } from '../../utils';
import { getPairsFromObject } from '../../webhook/utils';

import classes from './settings-list.module.scss';

type SettingsListProps = {
    sink: SinkConfig;
};

const OUTPUT_FORMAT_TYPE_LABEL_KEYS = {
    [OutputFormat.IMAGE_ORIGINAL]: 'inference.sinks.settings.outputFormatTypes.imageOriginal',
    [OutputFormat.IMAGE_WITH_PREDICTIONS]: 'inference.sinks.settings.outputFormatTypes.imageWithPredictions',
    [OutputFormat.PREDICTIONS]: 'inference.sinks.settings.outputFormatTypes.predictions',
} as const;

const OutputFormats = ({ outputFormats }: { outputFormats: SinkOutputFormats }) => {
    const { t } = useTranslation();

    return (
        <ul>
            {outputFormats.map((item) => (
                <li key={item}>{t(OUTPUT_FORMAT_TYPE_LABEL_KEYS[item])}</li>
            ))}
        </ul>
    );
};

const WebhookHeaders = ({ sink }: { sink: WebhookSinkConfig }) => {
    return (
        <ul>
            {getPairsFromObject(sink.headers ?? {}).map((pair) => (
                <li key={pair.key}>
                    {pair.key}: {pair.value}
                </li>
            ))}
        </ul>
    );
};

export const SettingsList = ({ sink }: SettingsListProps) => {
    const { t } = useTranslation();

    if (sink.sink_type === 'folder') {
        return (
            <ul className={classes.list}>
                <li>{t('inference.sinks.settings.folderPath', { path: sink.folder_path })}</li>
                <li>{t('inference.sinks.settings.rateLimit', { value: formatRateLimit(sink.rate_limit, t) })}</li>
                <li>
                    {t('inference.sinks.settings.outputFormats')}
                    <OutputFormats outputFormats={sink.output_formats} />
                </li>
            </ul>
        );
    }

    if (sink.sink_type === 'webhook') {
        return (
            <ul className={classes.list}>
                <li>{t('inference.sinks.settings.rateLimit', { value: formatRateLimit(sink.rate_limit, t) })}</li>
                <li>{t('inference.sinks.settings.httpMethod', { value: sink.http_method })}</li>
                <li>{t('inference.sinks.settings.timeout', { value: sink.timeout })}</li>
                <li>{t('inference.sinks.settings.webhookUrl', { value: sink.webhook_url })}</li>
                <li>
                    {t('inference.sinks.fields.headers')} <WebhookHeaders sink={sink} />
                </li>
                <li>
                    {t('inference.sinks.settings.outputFormats')}
                    <OutputFormats outputFormats={sink.output_formats} />
                </li>
            </ul>
        );
    }

    if (sink.sink_type === 'mqtt') {
        const authRequiredValue = sink.auth_required
            ? t('inference.sinks.settings.yes')
            : t('inference.sinks.settings.no');

        return (
            <ul className={classes.list}>
                <li>{t('inference.sinks.settings.topic', { value: sink.topic })}</li>
                <li>{t('inference.sinks.settings.rateLimit', { value: formatRateLimit(sink.rate_limit, t) })}</li>
                <li>{t('inference.sinks.settings.authRequired', { value: authRequiredValue })}</li>
                <li>{t('inference.sinks.settings.brokerHost', { value: sink.broker_host })}</li>
                <li>{t('inference.sinks.settings.brokerPort', { value: sink.broker_port })}</li>
                <li>
                    {t('inference.sinks.settings.outputFormats')}
                    <OutputFormats outputFormats={sink.output_formats} />
                </li>
            </ul>
        );
    }

    if (sink.sink_type === 'ros') {
        return (
            <ul className={classes.list}>
                <li>{t('inference.sinks.settings.topic', { value: sink.topic })}</li>
                <li>{t('inference.sinks.settings.rateLimit', { value: formatRateLimit(sink.rate_limit, t) })}</li>
                <li>
                    {t('inference.sinks.settings.outputFormats')}
                    <OutputFormats outputFormats={sink.output_formats} />
                </li>
            </ul>
        );
    }

    return <></>;
};
