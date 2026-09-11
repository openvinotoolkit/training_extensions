// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import type { LocalFolderSinkConfig, MqttSinkConfig, WebhookSinkConfig } from '@/api/types';
import { useTranslation } from '@/i18n';

import { ReactComponent as FolderIcon } from '../../../assets/icons/folder.svg';
import { ReactComponent as MqttIcon } from '../../../assets/icons/mqtt.svg';
import { ReactComponent as WebhookIcon } from '../../../assets/icons/webhook.svg';
import { DisclosureGroup } from '../sources/disclosure-group.component';
import { AddSink } from './add-sink/add-sink.component';
import { LocalFolder } from './local-folder/local-folder.component';
import { getLocalFolderInitialConfig, localFolderBodyFormatter } from './local-folder/utils';
import { Mqtt } from './mqtt/mqtt.component';
import { getMqttInitialConfig, mqttBodyFormatter } from './mqtt/utils';
import { getWebhookInitialConfig, webhookBodyFormatter } from './webhook/utils';
import { Webhook } from './webhook/webhook.component';

interface SinkOptionsProps {
    onSaved: () => void;
    hasHeader: boolean;
    children: ReactNode;
}

export const SinkOptions = ({ hasHeader, onSaved, children }: SinkOptionsProps) => {
    const { t } = useTranslation();

    return (
        <>
            {hasHeader && children}
            <DisclosureGroup
                defaultActiveInput={null}
                items={[
                    {
                        label: t('inference.sinks.options.folder'),
                        value: 'folder',
                        icon: <FolderIcon width={'24px'} />,
                        content: (
                            <AddSink
                                onSaved={onSaved}
                                config={getLocalFolderInitialConfig()}
                                componentFields={(state: LocalFolderSinkConfig) => <LocalFolder defaultState={state} />}
                                bodyFormatter={localFolderBodyFormatter}
                            />
                        ),
                    },
                    {
                        label: t('inference.sinks.options.webhook'),
                        value: 'webhook',
                        icon: <WebhookIcon width={'24px'} />,
                        content: (
                            <AddSink
                                onSaved={onSaved}
                                config={getWebhookInitialConfig()}
                                componentFields={(state: WebhookSinkConfig) => <Webhook defaultState={state} />}
                                bodyFormatter={webhookBodyFormatter}
                            />
                        ),
                    },
                    {
                        label: t('inference.sinks.options.mqtt'),
                        value: 'mqtt',
                        icon: <MqttIcon width={'24px'} />,
                        content: (
                            <AddSink
                                onSaved={onSaved}
                                config={getMqttInitialConfig()}
                                componentFields={(state: MqttSinkConfig) => <Mqtt defaultState={state} />}
                                bodyFormatter={mqttBodyFormatter}
                            />
                        ),
                    },
                ]}
            />
        </>
    );
};
