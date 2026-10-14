// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { Loading } from '@geti-ui/ui';

import { JobActionsMenu } from './job-actions-menu.component';
import { JobRowLayout, type JobRowProps } from './job-row-layout.component';

export const QueuedJobRow = ({ job, showStatusBadge, projectName, architectureName }: JobRowProps) => {
    const { t } = useTranslation();

    return (
        <JobRowLayout
            job={job}
            showStatusBadge={showStatusBadge}
            projectName={projectName}
            architectureName={architectureName}
            meta={t('models.jobs.waitingToStart')}
            spinner={<Loading size={'S'} mode={'inline'} />}
            actions={<JobActionsMenu job={job} canCancel />}
        />
    );
};
