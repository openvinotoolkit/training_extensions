// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { QuantizeJob, TrainJob } from '@/api/types';
import { useTranslation } from '@/i18n';
import { Badge, Button } from '@geti-ui/ui';

import { JobRow, type JobRowColumnsProps } from './job-row.component';

type FailedJobRowProps = JobRowColumnsProps & {
    job: TrainJob | QuantizeJob;
    onDismiss?: () => void;
};

const FailedStatusBadge = () => {
    const { t } = useTranslation();

    return <Badge variant={'negative'}>{t('models.jobs.failed')}</Badge>;
};

const DismissFailedJob = ({ onDismiss }: { onDismiss: () => void }) => {
    const { t } = useTranslation();

    return (
        <Button variant={'negative'} onPress={onDismiss} aria-label={'Dismiss failed job'}>
            {t('models.jobs.dismiss')}
        </Button>
    );
};

// The failed job's error is not shown here, it can be an arbitrarily long traceback; the logs dialog has it
export const FailedJobRow = ({ job, onDismiss, groupBy, datasetRevisions, modelArchitectures }: FailedJobRowProps) => {
    return (
        <JobRow
            job={job}
            progress={0}
            statusBadges={<FailedStatusBadge />}
            actions={onDismiss && <DismissFailedJob onDismiss={onDismiss} />}
            groupBy={groupBy}
            datasetRevisions={datasetRevisions}
            modelArchitectures={modelArchitectures}
        />
    );
};
