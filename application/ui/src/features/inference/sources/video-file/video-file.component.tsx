// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useRef, useState } from 'react';

import type { VideoFileSourceConfig } from '@/api/types';
import { UploadProgress } from '@/components/upload-progress/upload-progress.component';
import { useTranslation } from '@/i18n';
import { Button, Flex, Switch, Text, TextField } from '@geti-ui/ui';

import { acceptedVideoExtensions } from '../../../../shared/media-file-utils';
import type { VideoFileUploadProgress } from './use-video-file-upload.hook';

import classes from './video-file.module.scss';

type VideoFileProps = {
    defaultState?: VideoFileSourceConfig;
    /** Progress of the selected file's transfer while the source is being saved. */
    uploadProgress?: VideoFileUploadProgress | null;
    onCancelUpload?: () => void;
};

const ACCEPTED_VIDEO_EXTENSIONS = [acceptedVideoExtensions, '.flv', '.wmv', '.mpg', '.mpeg'].join(',');

export const VideoFile = ({ defaultState, uploadProgress, onCancelUpload }: VideoFileProps) => {
    const { t } = useTranslation();
    const [videoPath, setVideoPath] = useState(defaultState?.video_path ?? '');
    const [selectedFile, setSelectedFile] = useState<File | null>(null);

    const fileInputRef = useRef<HTMLInputElement>(null);

    // Typing a path and choosing a file are mutually exclusive; the most recent action wins.
    const handlePathChange = (value: string) => {
        setVideoPath(value);
        setSelectedFile(null);

        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }
    };

    const handleFileChange = (file: File | null) => {
        setSelectedFile(file);

        if (file !== null) {
            setVideoPath('');
        }
    };

    return (
        <Flex direction='column' gap='size-200'>
            <TextField isHidden label='id' name='id' defaultValue={defaultState?.id} />
            <TextField width='100%' label={t('common.labels.name')} name='name' defaultValue={defaultState?.name} />

            <Flex direction='column' gap='size-100'>
                <Flex gap='size-100' alignItems='end'>
                    <TextField
                        isRequired={selectedFile === null}
                        flex='1'
                        name='video_path'
                        label={t('inference.sources.fields.videoFilePath')}
                        value={videoPath}
                        onChange={handlePathChange}
                    />

                    <input
                        ref={fileInputRef}
                        type='file'
                        name='video_file'
                        hidden
                        data-testid='upload-video-file'
                        accept={ACCEPTED_VIDEO_EXTENSIONS}
                        onChange={(event) => handleFileChange(event.target.files?.[0] ?? null)}
                    />
                    <Button variant='secondary' onPress={() => fileInputRef.current?.click()}>
                        {t('common.actions.upload')}
                    </Button>
                </Flex>

                {selectedFile !== null && (
                    <Flex alignItems='center' gap='size-100'>
                        <Text UNSAFE_className={classes.selectedRow}>
                            {t('inference.sources.fields.selectedFile', { fileName: selectedFile.name })}
                        </Text>
                    </Flex>
                )}

                {uploadProgress && (
                    <UploadProgress
                        bytesSent={uploadProgress.bytesSent}
                        bytesTotal={uploadProgress.bytesTotal}
                        // Once the transfer is complete, the server is already consuming the upload: cancelling the
                        // signal at that point would no longer prevent the source from being saved.
                        onCancel={uploadProgress.bytesSent < uploadProgress.bytesTotal ? onCancelUpload : undefined}
                    />
                )}
            </Flex>

            <Switch
                aria-label='loop video'
                name='loop'
                defaultSelected={defaultState?.loop}
                key={defaultState?.loop ? 'true' : 'false'}
            >
                {t('inference.sources.fields.loopVideo')}
            </Switch>
        </Flex>
    );
};
