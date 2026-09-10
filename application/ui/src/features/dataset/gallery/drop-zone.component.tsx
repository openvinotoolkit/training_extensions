// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { useTranslation } from '@/i18n';
import {
    AriaDropZone,
    Content,
    Flex,
    Heading,
    IllustratedMessage,
    Text,
    View,
    type SpectrumDropZoneProps,
} from '@geti-ui/ui';

import { ReactComponent as DropFiles } from '../../../assets/drop-files.svg';
import { getFilesFromDropEvent } from '../../../shared/drop-zone.utils';

import classes from './drop-zone.component.module.scss';

type DatasetDropZoneProps = {
    children: ReactNode;
    onFilesDropped: (files: File[]) => void | Promise<void>;
};

type DropEvent = Parameters<NonNullable<SpectrumDropZoneProps['onDrop']>>[0];

export const DatasetDropZone = ({ children, onFilesDropped }: DatasetDropZoneProps) => {
    const { t } = useTranslation();

    const handleDrop = async (event: DropEvent) => {
        const files = await getFilesFromDropEvent(event);

        onFilesDropped(files);
    };

    return (
        <AriaDropZone onDrop={handleDrop} className={classes.dropZone}>
            {(dropZoneState) => (
                <View UNSAFE_className={classes.container}>
                    {children}

                    {dropZoneState.isDropTarget && (
                        <Flex alignItems={'center'} justifyContent={'center'} UNSAFE_className={classes.dropOverlay}>
                            <IllustratedMessage UNSAFE_className={classes.dropSurface}>
                                <DropFiles />

                                <Content>
                                    <Flex alignItems={'center'} direction={'column'} gap={'size-100'}>
                                        <Heading level={2}>{t('dataset.gallery.dropZone.title')}</Heading>
                                        <Text UNSAFE_className={classes.dropMessage}>
                                            {t('dataset.gallery.dropZone.description')}
                                        </Text>
                                    </Flex>
                                </Content>
                            </IllustratedMessage>
                        </Flex>
                    )}
                </View>
            )}
        </AriaDropZone>
    );
};
