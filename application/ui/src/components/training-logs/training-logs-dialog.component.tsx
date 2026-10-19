// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

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

import { downloadBlob } from '../../platform/download-file';
import { useDownloadModelLogs, useModelLogs } from './hooks/use-model-logs.hook';
import { useStreamJobLogs } from './hooks/use-stream-job-logs.hook';
import { LogViewer } from './log-viewer.component';

import classes from './training-logs-dialog.module.scss';

type TrainingLogsDialogProps = {
    projectId: string;
    isJobActive: boolean;
    jobId?: string;
    modelId?: string;
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
    const { downloadModelLogs, isDownloading } = useDownloadModelLogs(projectId, String(modelId));
    const { logs: jobLogs, connectionStatus } = useStreamJobLogs(jobId, { retry: isJobActive });

    const downloadJobLogs = () => {
        const content = jobLogs.map((entry) => JSON.stringify(entry)).join('\n');

        downloadBlob(
            new Blob([content], { type: 'application/x-ndjson' }),
            `job-logs-${jobId}.log`,
            t('models.training.logs.downloadStarted')
        );
    };

    return (
        <Dialog aria-label={'Training logs'} UNSAFE_className={classes.dialog}>
            <Heading>{t('models.training.logs.dialogTitle')}</Heading>
            <Header>
                <Flex alignItems={'center'} gap={'size-100'} marginStart={'auto'}>
                    {jobId && (
                        <ActionButton
                            isQuiet
                            onPress={downloadJobLogs}
                            aria-label={'Download logs'}
                            isDisabled={jobLogs.length === 0}
                        >
                            <DownloadIcon />
                        </ActionButton>
                    )}
                    {!jobId && modelId && (
                        <ActionButton
                            isQuiet
                            onPress={downloadModelLogs}
                            aria-label={'Download logs'}
                            isDisabled={isDownloading}
                        >
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
                {jobId && <LogViewer logs={jobLogs} isStreaming connectionStatus={connectionStatus} />}
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
