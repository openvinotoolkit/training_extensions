// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { PipelineComponentsHealth, PipelineStatus } from '@/api/types';
import type { StatusLightProps } from '@geti-ui/ui';
import { capitalize } from 'lodash-es';

type StatusVariant = StatusLightProps['variant'];

type ComponentStatusMeta = {
    label: string;
    variant: StatusVariant;
    message: string | null | undefined;
};

type StatusMeta = {
    label: string;
    variant: StatusVariant;
};

export const getOverallStatusMeta = (status: string): StatusMeta => {
    switch (status) {
        case 'running':
            return { label: 'Running', variant: 'positive' };
        case 'idle':
            return { label: 'Idle', variant: 'neutral' };
        case 'error':
            return { label: 'Problems detected', variant: 'negative' };
        default:
            return { label: capitalize(status), variant: 'neutral' };
    }
};

export const getComponentStatusMeta = (component: PipelineStatus): ComponentStatusMeta => {
    switch (component.status) {
        case 'ok':
            return { label: 'Healthy', variant: 'positive', message: component.message };
        case 'finished':
            return { label: 'Finished', variant: 'info', message: component.message };
        case 'unavailable':
            return { label: 'Unavailable', variant: 'neutral', message: component.message };
        case 'error':
            return { label: 'Error', variant: 'negative', message: component.message };
        default:
            return { label: capitalize(component.status), variant: 'neutral', message: component.message };
    }
};

export const shouldShowPipelineHealthDetails = (components: PipelineComponentsHealth | null | undefined): boolean => {
    if (components == null) {
        return false;
    }

    return [components.source, components.sink, components.model].some((component) => component.message != null);
};
