// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { Flex, NumberField, Text } from '@geti-ui/ui';

import { positiveNumberOrUndefined } from '../utils';

type RateLimitFieldsProps = {
    rateLimit: number | null | undefined;
};

export const RateLimitFields = ({ rateLimit }: RateLimitFieldsProps) => {
    const { t } = useTranslation();
    const samples = positiveNumberOrUndefined(rateLimit);
    const seconds = samples !== undefined ? 1 : undefined;

    return (
        <Flex gap='size-100' alignItems={'end'} wrap>
            <NumberField
                label={t('inference.sinks.fields.samples')}
                name='rate_limit_samples'
                minValue={0.1}
                step={0.1}
                defaultValue={samples}
            />
            <Text>{t('inference.sinks.fields.every')}</Text>
            <NumberField
                label={t('inference.sinks.fields.seconds')}
                name='rate_limit_seconds'
                minValue={0.1}
                step={0.1}
                defaultValue={seconds}
            />
        </Flex>
    );
};
