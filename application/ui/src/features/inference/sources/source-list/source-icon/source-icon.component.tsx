// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import {
    GenICam as GenICamIcon,
    ImagesFolder as ImagesFolderIcon,
    IpCamera as IpCameraIcon,
    VideoFile as VideoFileIcon,
    Webcam as WebcamIcon,
} from '@/assets/icons';

type SourceIconProps = {
    type: string;
};

export const SourceIcon = ({ type }: SourceIconProps) => {
    if (type === 'usb_camera') {
        return <WebcamIcon />;
    }

    if (type === 'ip_camera') {
        return <IpCameraIcon />;
    }

    if (type === 'video_file') {
        return <VideoFileIcon />;
    }

    if (type === 'gen_i_cam') {
        return <GenICamIcon />;
    }

    return <ImagesFolderIcon />;
};
