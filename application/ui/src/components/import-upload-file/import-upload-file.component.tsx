// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useRef, useState } from 'react';

import { isAbortError, uploadDatasetArchive } from '@/api';
import { DropFiles } from '@/assets/illustrations';
import { useTranslation } from '@/i18n';
import { Button, Content, DropZone, FileTrigger, Flex, Heading, IllustratedMessage, Text } from '@geti-ui/ui';
import { LinkOut } from '@geti-ui/ui/icons';
import { useMutation } from '@tanstack/react-query';
import { useSubmitJob } from 'hooks/api/jobs/jobs.hook';

import { Link } from '../../platform/components/link.component';
import { getFilesFromDropEvent } from '../../shared/drop-zone.utils';
import { ThreeDotsFlashing } from '../three-dots-flashing/three-dots-flashing.component';
import { toast } from '../toast/toast.component';
import { UploadProgress } from '../upload-progress/upload-progress.component';
import { formatToFileArray, isSupportedDatasetZip } from './util';

import classes from './import-upload-file.module.scss';

export type FileUploadedResponse = { size: number; fileName: string; prepareJobId: string; stagedDatasetId: string };

type ImportUploadFileProps = {
    formatOptions: string;
    onFileUploaded: (data: FileUploadedResponse) => void;
};

const useUploadFile = (onProgress: (bytesSent: number) => void) => {
    const abortControllerRef = useRef<AbortController | null>(null);
    const stagedDatasetMutation = useMutation({
        mutationFn: (file: File) => {
            const abortController = new AbortController();
            abortControllerRef.current = abortController;

            return uploadDatasetArchive(file, { onProgress, signal: abortController.signal });
        },
        onSettled: () => {
            abortControllerRef.current = null;
        },
        // A cancelled upload is a deliberate user action, not an error worth notifying about.
        meta: { error: { notify: (error: unknown) => !isAbortError(error) } },
    });

    // Closing the dialog mid-transfer discards the partial upload instead of finishing it in the background.
    useEffect(() => {
        return () => {
            abortControllerRef.current?.abort();
        };
    }, []);

    return { stagedDatasetMutation, abort: () => abortControllerRef.current?.abort() };
};

export const ImportUploadFile = ({ formatOptions, onFileUploaded }: ImportUploadFileProps) => {
    const { t } = useTranslation();
    const [bytesSent, setBytesSent] = useState(0);
    const [fileSize, setFileSize] = useState(0);
    const { stagedDatasetMutation, abort } = useUploadFile(setBytesSent);
    const prepareImportJobMutation = useSubmitJob();

    const handleLoadingFile = (files: File[]) => {
        const hasMultipleFiles = files.length > 1;

        if (hasMultipleFiles) {
            toast({
                message: t('dataset.import.multipleFilesError'),
                type: 'error',
            });
            return;
        }

        if (!isSupportedDatasetZip(files[0])) {
            toast({
                message: t('dataset.import.unsupportedFormatError'),
                type: 'error',
            });
            return;
        }

        // Failures are already reported by the mutations' error notifications.
        handleImportPrepare(files[0]).catch(() => undefined);
    };

    const handleImportPrepare = async (file: File) => {
        setBytesSent(0);
        setFileSize(file.size);
        const stagedDataset = await stagedDatasetMutation.mutateAsync(file);

        const prepareImportJob = await prepareImportJobMutation.mutateAsync({
            body: {
                job_type: 'prepare_dataset_for_import',
                staged_dataset_id: stagedDataset.id,
            },
        });

        onFileUploaded({
            size: file.size,
            fileName: file.name,
            prepareJobId: prepareImportJob.job_id,
            stagedDatasetId: stagedDataset.id,
        });
    };

    const isPending = stagedDatasetMutation.isPending || prepareImportJobMutation.isPending;

    return (
        <DropZone
            isFilled={stagedDatasetMutation.isSuccess}
            onDrop={async (event) => handleLoadingFile(await getFilesFromDropEvent(event))}
        >
            <IllustratedMessage maxHeight={'size-4600'}>
                <DropFiles />

                <Content>
                    {isPending && (
                        <Flex alignItems={'center'} direction={'column'} gap={'size-100'}>
                            <Heading level={1} UNSAFE_className={classes.statusTitle}>
                                {stagedDatasetMutation.isPending
                                    ? t('common.status.uploading')
                                    : t('dataset.import.preparingJob')}
                                <ThreeDotsFlashing />
                            </Heading>
                            {stagedDatasetMutation.isPending && bytesSent < fileSize ? (
                                <>
                                    <Text>{t('dataset.import.datasetBeingUploaded')}</Text>
                                    <UploadProgress
                                        bytesSent={bytesSent}
                                        bytesTotal={fileSize}
                                        width={'size-4600'}
                                        onCancel={abort}
                                    />
                                </>
                            ) : (
                                <Text>{t('dataset.import.scanningMessage')}</Text>
                            )}
                        </Flex>
                    )}

                    {!isPending && (
                        <Flex alignItems={'center'} direction={'column'} gap={'size-100'}>
                            <Text>{t('dataset.import.dropZipHere')}</Text>

                            <FileTrigger
                                data-testid='upload-zip-file'
                                onSelect={(data) => handleLoadingFile(formatToFileArray(data))}
                            >
                                <Button marginY={'size-200'} maxWidth={'size-1000'} variant={'accent'}>
                                    {t('common.actions.upload')}
                                </Button>
                            </FileTrigger>

                            <Text UNSAFE_className={classes.formatOptions}>({formatOptions}).zip</Text>

                            <Link href='/' target='_blank' rel='noopener noreferrer' UNSAFE_className={classes.link}>
                                {t('dataset.import.learnMoreFormats')}
                                <LinkOut size='XS' />
                            </Link>
                        </Flex>
                    )}
                </Content>
            </IllustratedMessage>
        </DropZone>
    );
};
