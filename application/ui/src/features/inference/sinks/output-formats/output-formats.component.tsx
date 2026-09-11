// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { SinkOutputFormats } from '@/api/types';
import { useTranslation } from '@/i18n';
import { Checkbox, CheckboxGroup } from '@geti-ui/ui';

import { OutputFormat } from '../utils';

import classes from './output-formats.module.scss';

type OutputFormatsProps = {
    config?: SinkOutputFormats;
};

export const OutputFormats = ({ config = [] }: OutputFormatsProps) => {
    const { t } = useTranslation();

    return (
        <CheckboxGroup
            isRequired
            label={t('inference.sinks.fields.outputFormatsLabel')}
            name='output_formats'
            defaultValue={config}
            UNSAFE_className={classes.itemList}
        >
            <Checkbox name='output_formats' value={OutputFormat.PREDICTIONS}>
                {t('inference.sinks.fields.outputFormatPredictions')}
            </Checkbox>
            <Checkbox name='output_formats' value={OutputFormat.IMAGE_ORIGINAL}>
                {t('inference.sinks.fields.outputFormatImageOriginal')}
            </Checkbox>
            <Checkbox name='output_formats' value={OutputFormat.IMAGE_WITH_PREDICTIONS}>
                {t('inference.sinks.fields.outputFormatImageWithPredictions')}
            </Checkbox>
        </CheckboxGroup>
    );
};
