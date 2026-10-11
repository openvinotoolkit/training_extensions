// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Folder as FolderIcon, Mqtt as MqttIcon, Ros as RosIcon, Webhook as WebhookIcon } from '@/assets/icons';

interface SinkIconProps {
    type: 'folder' | 'mqtt' | 'webhook' | 'ros' | 'disconnected';
}

export const SinkIcon = ({ type }: SinkIconProps) => {
    if (type === 'folder') {
        return <FolderIcon />;
    }

    if (type === 'mqtt') {
        return <MqttIcon />;
    }

    if (type === 'ros') {
        return <RosIcon />;
    }

    if (type === 'webhook') {
        return <WebhookIcon />;
    }

    return <></>;
};
