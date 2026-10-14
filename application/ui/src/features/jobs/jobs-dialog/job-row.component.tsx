// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { getStatusGroup } from '../utils';
import { CancelledJobRow } from './cancelled-job-row.component';
import { FailedJobRow } from './failed-job-row.component';
import { FinishedJobRow } from './finished-job-row.component';
import { JOB_GRID_COLUMNS, JobRowLayout, type JobRowProps } from './job-row-layout.component';
import { QueuedJobRow } from './queued-job-row.component';
import { RunningJobRow } from './running-job-row.component';

export { JOB_GRID_COLUMNS, JobRowLayout };
export type { JobRowProps } from './job-row-layout.component';

export const JobRow = (props: JobRowProps) => {
    switch (getStatusGroup(props.job.status)) {
        case 'running':
            return <RunningJobRow {...props} />;
        case 'scheduled':
            return <QueuedJobRow {...props} />;
        case 'finished':
            return <FinishedJobRow {...props} />;
        case 'failed':
            return <FailedJobRow {...props} />;
        case 'cancelled':
            return <CancelledJobRow {...props} />;
    }
};
