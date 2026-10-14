// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { BottomProgressBar } from '@/components/bottom-progress-bar/bottom-progress-bar.component';
import { useTranslation } from '@/i18n';
import { Loading } from '@geti-ui/ui';
import { DATE_TIME_FORMAT, useFormatDate } from 'hooks/use-format-date.hook';

import { formatElapsed } from '../../utils';
import { JobActionsMenu } from './job-actions-menu.component';
import { JobRowLayout, type JobRowProps } from './job-row-layout.component';

export const RunningJobRow = ({ job, showStatusBadge, projectName, architectureName, onViewLogs }: JobRowProps) => {
    const { t } = useTranslation();
    const formatDateTime = useFormatDate(DATE_TIME_FORMAT);
    const startedAt = formatDateTime(job.started_at);
    const meta = [
        startedAt && t('jobs.meta.started', { time: startedAt }),
        t('jobs.meta.elapsed', { time: formatElapsed(job) }),
        job.progress > 0 && `${Math.round(job.progress)}%`,
    ]
        .filter(Boolean)
        .join(' · ');

    return (
        <BottomProgressBar progress={job.progress} color='var(--energy-blue)' ariaLabel={'Job progress'}>
            <JobRowLayout
                job={job}
                showStatusBadge={showStatusBadge}
                projectName={projectName}
                architectureName={architectureName}
                meta={meta}
                spinner={job.progress === 0 ? <Loading size={'S'} mode={'inline'} /> : undefined}
                actions={<JobActionsMenu job={job} canCancel={job.status === 'RUNNING'} onViewLogs={onViewLogs} />}
            />
        </BottomProgressBar>
    );
};
