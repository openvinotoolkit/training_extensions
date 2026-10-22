// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { API_BASE_URL } from '@/api';
import { useTranslation } from '@/i18n';
import {
    ActionButton,
    Content,
    Dialog,
    Divider,
    Flex,
    Header,
    Heading,
    Loading,
    Text,
    useDialogContainer,
} from '@geti-ui/ui';
import { CloseSemiBold, DownloadIcon } from '@geti-ui/ui/icons';

import { downloadFile } from '../../platform/download-file';
import { useModelLogs } from './hooks/use-model-logs.hook';
import { useStreamJobLogs } from './hooks/use-stream-job-logs.hook';
import { LogViewer } from './log-viewer.component';

import classes from './training-logs-dialog.module.scss';

type TrainingLogsDialogProps = {
    projectId: string;
    isJobActive: boolean;
    jobId?: string;
    modelId?: string;
};

const ActiveJobLogs = ({ jobId, isJobActive }: { jobId: string; isJobActive: boolean }) => {
    const { logs, connectionStatus } = useStreamJobLogs(jobId, { retry: isJobActive });

    return <LogViewer logs={logs} isStreaming connectionStatus={connectionStatus} />;
};

const HistoricalModelLogs = ({ projectId, modelId }: { projectId: string; modelId: string }) => {
    const { t } = useTranslation();
    const { data: logs, isPending, isError, error } = useModelLogs(projectId, modelId);

    if (isPending) {
        return (
            <Flex alignItems={'center'} justifyContent={'center'} height={'100%'}>
                <Loading size={'M'} />
            </Flex>
        );
    }

    if (isError) {
        return (
            <Flex alignItems={'center'} justifyContent={'center'} height={'100%'}>
                <Text UNSAFE_className={classes.errorText}>
                    {t('models.training.logs.loadError', {
                        message: error?.message ?? t('common.errors.unknownDetail'),
                    })}
                </Text>
            </Flex>
        );
    }

    return <LogViewer logs={logs ?? []} />;
};

export const TrainingLogsDialog = ({ jobId, modelId, projectId, isJobActive }: TrainingLogsDialogProps) => {
    const { t } = useTranslation();
    const dialogContainer = useDialogContainer();

    const downloadLogs = () => {
        const [path, name] = jobId
            ? [`/api/jobs/${jobId}/logs`, `job-logs-${jobId}.log`]
            : [`/api/projects/${projectId}/models/${modelId}/logs`, `training-logs-${modelId}.log`];

        downloadFile(`${API_BASE_URL}${path}?format=text`, name, t('models.training.logs.downloadStarted'));
    };

    return (
        <Dialog aria-label={'Training logs'} UNSAFE_className={classes.dialog}>
            <Heading>{t('models.training.logs.dialogTitle')}</Heading>
            <Header>
                <Flex alignItems={'center'} gap={'size-100'} marginStart={'auto'}>
                    {(jobId || modelId) && (
                        <ActionButton isQuiet onPress={downloadLogs} aria-label={'Download logs'}>
                            <DownloadIcon />
                        </ActionButton>
                    )}
                    <ActionButton
                        isQuiet
                        onPress={dialogContainer.dismiss}
                        aria-label={'Close dialog'}
                        UNSAFE_className={classes.closeButton}
                    >
                        <CloseSemiBold width={14} height={14} />
                    </ActionButton>
                </Flex>
            </Header>
            <Divider />
            <Content UNSAFE_className={classes.contentArea}>
                {jobId && <ActiveJobLogs jobId={jobId} isJobActive={isJobActive} />}
                {!jobId && modelId && <HistoricalModelLogs projectId={projectId} modelId={modelId} />}
                {!jobId && !modelId && (
                    <Flex alignItems={'center'} justifyContent={'center'} height={'100%'}>
                        <Text UNSAFE_className={classes.errorText}>{t('models.training.logs.noJobOrModel')}</Text>
                    </Flex>
                )}
            </Content>
        </Dialog>
    );
};
