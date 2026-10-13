// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { ImagesFolderSourceConfig, SourceConfig, VideoFileSourceConfig } from '@/api/types';

import { EditSource } from './edit-source/edit-source.component';
import { ImageFolder } from './image-folder/image-folder.component';
import { imagesFolderBodyFormatter } from './image-folder/utils';
import { IpCamera } from './ip-camera/ip-camera.component';
import { ipCameraBodyFormatter } from './ip-camera/utils';
import { UsbCamera } from './usb-camera/usb-camera-fields.component';
import { usbCameraBodyFormatter } from './usb-camera/utils';
import { useVideoFileUpload } from './video-file/use-video-file-upload.hook';
import { videoFileBodyFormatter } from './video-file/utils';
import { VideoFile } from './video-file/video-file.component';

interface EditSourceFormProps {
    config: SourceConfig;
    connectedSourceId: string | undefined;
    onSaved: () => void;
    onBackToList: () => void;
}

export const EditSourceForm = ({ config, connectedSourceId, onSaved, onBackToList }: EditSourceFormProps) => {
    const videoFileUpload = useVideoFileUpload();

    if (config.source_type === 'usb_camera') {
        return (
            <EditSource
                onSaved={onSaved}
                config={config}
                onBackToList={onBackToList}
                componentFields={(state) => <UsbCamera defaultState={state} />}
                bodyFormatter={usbCameraBodyFormatter}
                isConnected={connectedSourceId === config.id}
            />
        );
    }

    if (config.source_type === 'ip_camera') {
        return (
            <EditSource
                onSaved={onSaved}
                config={config}
                onBackToList={onBackToList}
                componentFields={(state) => <IpCamera defaultState={state} />}
                bodyFormatter={ipCameraBodyFormatter}
                isConnected={connectedSourceId === config.id}
            />
        );
    }

    if (config.source_type === 'video_file') {
        return (
            <EditSource
                onSaved={onSaved}
                config={config}
                onBackToList={onBackToList}
                componentFields={(state: VideoFileSourceConfig) => (
                    <VideoFile
                        defaultState={state}
                        uploadProgress={videoFileUpload.progress}
                        onCancelUpload={videoFileUpload.cancel}
                    />
                )}
                bodyFormatter={videoFileBodyFormatter}
                prepareFormData={videoFileUpload.prepareFormData}
                isConnected={connectedSourceId === config.id}
            />
        );
    }

    return (
        <EditSource
            onSaved={onSaved}
            onBackToList={onBackToList}
            config={config as ImagesFolderSourceConfig}
            componentFields={(state: ImagesFolderSourceConfig) => <ImageFolder defaultState={state} />}
            bodyFormatter={imagesFolderBodyFormatter}
            isConnected={connectedSourceId === config.id}
        />
    );
};
