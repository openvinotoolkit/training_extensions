// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { $api } from '@/api';
import type { SourceConfig } from '@/api/types';
import { useTranslation } from '@/i18n';

import classes from './settings-list.module.scss';

interface SettingsListProps {
    source: SourceConfig;
}

const CameraDeviceDisplay = ({ deviceId }: { deviceId: number }) => {
    const { t } = useTranslation();
    const { data: cameraDevices = [], isLoading } = $api.useQuery('get', '/api/system/devices/camera');
    const device = cameraDevices.find(({ index }) => index === deviceId);

    if (isLoading) {
        return <span>{t('inference.sources.settings.loading')}</span>;
    }

    const deviceLabel = device ? device.name : t('inference.sources.settings.unknownDevice', { deviceId });

    return (
        <ul className={classes.list}>
            <li>{t('inference.sources.settings.device', { name: deviceLabel })}</li>
        </ul>
    );
};

export const SettingsList = ({ source }: SettingsListProps) => {
    const { t } = useTranslation();

    if (source.source_type === 'images_folder') {
        const ignoreExistingImagesValue = source.ignore_existing_images
            ? t('inference.sources.settings.yes')
            : t('inference.sources.settings.no');

        return (
            <ul className={classes.list}>
                <li>{t('inference.sources.settings.folderPath', { path: source.images_folder_path })}</li>
                <li>{t('inference.sources.settings.ignoreExistingImages', { value: ignoreExistingImagesValue })}</li>
            </ul>
        );
    }

    if (source.source_type === 'ip_camera') {
        const authRequiredValue = source.auth_required
            ? t('inference.sources.settings.yes')
            : t('inference.sources.settings.no');

        return (
            <ul className={classes.list}>
                <li>{t('inference.sources.settings.streamUrl', { url: source.stream_url })}</li>
                <li>{t('inference.sources.settings.authRequired', { value: authRequiredValue })}</li>
            </ul>
        );
    }

    if (source.source_type === 'video_file') {
        return (
            <ul className={classes.list}>
                <li>{t('inference.sources.settings.videoPath', { path: source.video_path })}</li>
            </ul>
        );
    }

    if (source.source_type === 'usb_camera') {
        return <CameraDeviceDisplay deviceId={source.device_id} />;
    }

    return <></>;
};
