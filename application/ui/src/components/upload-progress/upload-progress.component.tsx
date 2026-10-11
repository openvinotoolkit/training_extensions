// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { Button, Flex, ProgressBar, Text, type DimensionValue } from '@geti-ui/ui';

import { formatBytes } from '../../shared/util';

type UploadProgressProps = {
    bytesSent: number;
    bytesTotal: number;
    /** Renders a cancel button when provided. */
    onCancel?: () => void;
    width?: DimensionValue;
};

export const getUploadPercentage = (bytesSent: number, bytesTotal: number): number =>
    bytesTotal > 0 ? Math.min(100, Math.round((bytesSent / bytesTotal) * 100)) : 0;

export const UploadProgress = ({ bytesSent, bytesTotal, onCancel, width = '100%' }: UploadProgressProps) => {
    const { t } = useTranslation();

    return (
        <Flex direction={'column'} gap={'size-50'} width={width}>
            <ProgressBar
                aria-label={'Upload progress'}
                value={getUploadPercentage(bytesSent, bytesTotal)}
                width={'100%'}
            />
            <Flex alignItems={'center'} justifyContent={'space-between'} gap={'size-100'}>
                <Text>
                    {t('common.labels.bytesTransferred', {
                        transferred: formatBytes(bytesSent),
                        total: formatBytes(bytesTotal),
                    })}
                </Text>
                {onCancel !== undefined && (
                    <Button variant={'secondary'} onPress={onCancel}>
                        {t('common.actions.cancel')}
                    </Button>
                )}
            </Flex>
        </Flex>
    );
};
