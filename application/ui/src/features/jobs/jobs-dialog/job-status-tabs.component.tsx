// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { Flex, Item, TabList, Tabs, Text } from '@geti-ui/ui';

import type { StatusGroup } from '../utils';
import { STATUS_BADGE, TONE, ToneBadge } from './job-status-badge.component';

type JobStatusTabsProps = {
    counts: Record<StatusGroup, number>;
    selected: StatusGroup;
    onChange: (group: StatusGroup) => void;
};

const TABS = [
    { group: 'all', labelKey: 'jobs.tabs.all', tone: TONE.blue },
    { group: 'running', labelKey: 'common.status.running', tone: STATUS_BADGE.RUNNING.tone },
    { group: 'finished', labelKey: 'jobs.tabs.finished', tone: STATUS_BADGE.DONE.tone },
    { group: 'scheduled', labelKey: 'jobs.status.pending', tone: STATUS_BADGE.PENDING.tone },
    { group: 'cancelled', labelKey: 'common.status.cancelled', tone: STATUS_BADGE.CANCELLED.tone },
    { group: 'failed', labelKey: 'common.status.failed', tone: STATUS_BADGE.FAILED.tone },
] as const;

export const JobStatusTabs = ({ counts, selected, onChange }: JobStatusTabsProps) => {
    const { t } = useTranslation();

    return (
        <Tabs
            aria-label='Job status'
            selectedKey={selected}
            onSelectionChange={(key) => onChange(key as StatusGroup)}
            UNSAFE_style={{ '--spectrum-tabs-selection-indicator-color': 'var(--energy-blue)' }}
        >
            <TabList>
                {TABS.map(({ group, labelKey, tone }) => {
                    const label = t(labelKey);
                    const count = counts[group];

                    return (
                        <Item key={group} textValue={label}>
                            <Flex alignItems={'center'} gap={'size-100'}>
                                <Text>{label}</Text>
                                {count > 0 && (
                                    <ToneBadge tone={group === selected ? tone : TONE.grey}>{count}</ToneBadge>
                                )}
                            </Flex>
                        </Item>
                    );
                })}
            </TabList>
        </Tabs>
    );
};
