// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { DATE_TIME_FORMAT, useFormatDate } from 'hooks/use-format-date.hook';

import { formatElapsed } from '../../utils';
import { JobActionsMenu } from './job-actions-menu.component';
import { JobRowLayout, type JobRowProps } from './job-row-layout.component';

export const FinishedJobRow = ({ job, showStatusBadge, projectName, architectureName, onViewLogs }: JobRowProps) => {
    const { t } = useTranslation();
    const formatDateTime = useFormatDate(DATE_TIME_FORMAT);
    const finishedAt = formatDateTime(job.finished_at);
    const meta = [
        t('jobs.meta.duration', { time: formatElapsed(job) }),
        finishedAt && t('jobs.meta.finished', { time: finishedAt }),
    ]
        .filter(Boolean)
        .join(' · ');

    return (
        <JobRowLayout
            job={job}
            showStatusBadge={showStatusBadge}
            projectName={projectName}
            architectureName={architectureName}
            meta={meta}
            actions={<JobActionsMenu job={job} canCancel={false} onViewLogs={onViewLogs} />}
        />
    );
};
