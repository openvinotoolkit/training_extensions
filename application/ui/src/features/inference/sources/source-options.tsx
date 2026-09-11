// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import type { IPCameraSourceConfig, USBCameraSourceConfig, VideoFileSourceConfig } from '@/api/types';
import { useTranslation } from '@/i18n';

import { ReactComponent as IpCameraIcon } from '../../../assets/icons/ip-camera.svg';
import { ReactComponent as Video } from '../../../assets/icons/video-file.svg';
import { ReactComponent as WebcamIcon } from '../../../assets/icons/webcam.svg';
import { AddSource } from './add-source/add-source.component';
import { DisclosureGroup } from './disclosure-group.component';
import { IpCamera } from './ip-camera/ip-camera.component';
import { getIpCameraInitialConfig, ipCameraBodyFormatter } from './ip-camera/utils';
import { UsbCamera } from './usb-camera/usb-camera-fields.component';
import { getUsbCameraInitialConfig, usbCameraBodyFormatter } from './usb-camera/utils';
import { getVideoFileInitialConfig, prepareVideoFileFormData, videoFileBodyFormatter } from './video-file/utils';
import { VideoFile } from './video-file/video-file.component';

interface SourceOptionsProps {
    onSaved: () => void;
    hasHeader: boolean;
    children: ReactNode;
    existingNames?: string[];
}

export const SourceOptions = ({ onSaved, hasHeader, children, existingNames = [] }: SourceOptionsProps) => {
    const { t } = useTranslation();

    return (
        <>
            {hasHeader && children}

            <DisclosureGroup
                defaultActiveInput={null}
                items={[
                    {
                        label: t('inference.sources.options.usbCamera'),
                        value: 'usb_camera',
                        icon: <WebcamIcon width={'24px'} />,
                        content: (
                            <AddSource
                                onSaved={onSaved}
                                config={getUsbCameraInitialConfig(t, existingNames)}
                                componentFields={(state: USBCameraSourceConfig) => <UsbCamera defaultState={state} />}
                                bodyFormatter={usbCameraBodyFormatter}
                            />
                        ),
                    },
                    {
                        label: t('inference.sources.options.ipCamera'),
                        value: 'ip_camera',
                        icon: <IpCameraIcon width={'24px'} />,
                        content: (
                            <AddSource
                                onSaved={onSaved}
                                config={getIpCameraInitialConfig(t, existingNames)}
                                componentFields={(state: IPCameraSourceConfig) => <IpCamera defaultState={state} />}
                                bodyFormatter={ipCameraBodyFormatter}
                            />
                        ),
                    },
                    // TODO: Reenable after MVP
                    // {
                    //     label: 'GenICam',
                    //     value: 'gen_i_cam',

                    //     icon: <GenICam width={'24px'} />,
                    //     content: (
                    //         <AddSource
                    //             onSaved={onSaved}
                    //             config={getImagesFolderInitialConfig()}
                    //             componentFields={(state: ImagesFolderSourceConfig) => (
                    //                 <ImageFolder defaultState={state} />
                    //             )}
                    //             bodyFormatter={imagesFolderBodyFormatter}
                    //         />
                    //     ),
                    // },
                    {
                        label: t('inference.sources.options.videoFile'),
                        value: 'video_file',
                        icon: <Video width={'24px'} />,

                        content: (
                            <AddSource
                                onSaved={onSaved}
                                config={getVideoFileInitialConfig(t, existingNames)}
                                componentFields={(state: VideoFileSourceConfig) => <VideoFile defaultState={state} />}
                                bodyFormatter={videoFileBodyFormatter}
                                prepareFormData={prepareVideoFileFormData}
                            />
                        ),
                    },
                    // TODO: Reenable after MVP
                    // {
                    //     label: 'Images folder',
                    //     value: 'images_folder',
                    //     icon: <Image width={'24px'} />,
                    //     content: (
                    //         <AddSource
                    //             onSaved={onSaved}
                    //             config={getImagesFolderInitialConfig()}
                    //             componentFields={(state: ImagesFolderSourceConfig) => (
                    //                 <ImageFolder defaultState={state} />
                    //             )}
                    //             bodyFormatter={imagesFolderBodyFormatter}
                    //         />
                    //     ),
                    // },
                ]}
            />
        </>
    );
};
