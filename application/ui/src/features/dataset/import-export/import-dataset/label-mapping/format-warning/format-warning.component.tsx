// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { AnnotationType } from '@/api/types';
import { TranslateFn, useTranslation } from '@/i18n';
import { Divider, Flex, Text } from '@geti-ui/ui';
import { Alert } from '@geti-ui/ui/icons';
import { isNil } from 'lodash-es';

import { useProject } from '../../../../../../hooks/api/project.hook';

import classes from './format-warning.module.scss';

type FormatWarningProps = {
    annotationType?: AnnotationType;
};

const getMessage = (t: TranslateFn, taskType: string, annotationType?: AnnotationType) => {
    if (annotationType === 'bounding_box' && taskType === 'instance_segmentation') {
        return t('dataset.import.formatWarning.boundingBoxToPolygon');
    }

    if (annotationType === 'polygon' && taskType === 'detection') {
        return t('dataset.import.formatWarning.polygonToBoundingBox');
    }

    return null;
};

export const FormatWarning = ({ annotationType }: FormatWarningProps) => {
    const { t } = useTranslation();
    const { data: selectedProject } = useProject();

    const message = getMessage(t, selectedProject?.task?.task_type, annotationType);

    if (isNil(annotationType) || isNil(message)) {
        return null;
    }

    return (
        <>
            <Divider size={'S'} marginY={'size-125'} />

            <Flex gap={'size-125'}>
                <div className={classes.iconContainer}>
                    <Alert width={24} height={24} />
                </div>

                <Text>{message}</Text>
            </Flex>
        </>
    );
};
