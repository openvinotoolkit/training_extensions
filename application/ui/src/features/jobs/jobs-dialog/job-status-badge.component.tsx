// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { ReactNode } from 'react';

import { useTranslation } from '@/i18n';
import { Badge } from '@geti-ui/ui';

import classes from './jobs-dialog.module.scss';

export type BadgeTone = {
    variant: 'positive' | 'negative' | 'neutral' | 'yellow' | 'purple';
    className?: string;
};

export const TONE = {
    blue: { variant: 'neutral', className: classes.toneEnergyBlue },
    green: { variant: 'positive' },
    red: { variant: 'negative' },
    orange: { variant: 'neutral', className: classes.toneOrange },
    purple: { variant: 'purple' },
    yellow: { variant: 'yellow' },
    grey: { variant: 'neutral' },
} satisfies Record<string, BadgeTone>;

export const STATUS_BADGE = {
    PENDING: { labelKey: 'jobs.status.pending', tone: TONE.purple },
    RUNNING: { labelKey: 'common.status.running', tone: TONE.blue },
    CANCELLING: { labelKey: 'jobs.status.cancelling', tone: TONE.yellow },
    DONE: { labelKey: 'jobs.status.done', tone: TONE.green },
    FAILED: { labelKey: 'common.status.failed', tone: TONE.red },
    CANCELLED: { labelKey: 'common.status.cancelled', tone: TONE.orange },
} as const;

const isKnownStatus = (status: string): status is keyof typeof STATUS_BADGE => Object.hasOwn(STATUS_BADGE, status);

export const ToneBadge = ({ tone, children }: { tone: BadgeTone; children: ReactNode }) => (
    <Badge variant={tone.variant} UNSAFE_className={tone.className}>
        {children}
    </Badge>
);

export const JobStatusBadge = ({ status }: { status: string }) => {
    const { t } = useTranslation();

    if (!isKnownStatus(status)) {
        return <ToneBadge tone={TONE.grey}>{status}</ToneBadge>;
    }

    const { labelKey, tone } = STATUS_BADGE[status];

    return <ToneBadge tone={tone}>{t(labelKey)}</ToneBadge>;
};
