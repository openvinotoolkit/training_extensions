// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { ExportDatasetMetadata } from '@/api/types';
import { useTranslation } from '@/i18n';
import { dimensionValue, Divider, Flex, Grid, Text } from '@geti-ui/ui';
import { isEmpty, isNil } from 'lodash-es';

import { useProject } from '../../../../../../hooks/api/project.hook';

type ExportJobDetailsProps = {
    datasetName?: string;
    metadata: ExportDatasetMetadata;
};

const isGetiFormat = (format?: string | null) => format?.toLowerCase() === 'geti';

export const ExportJobDetails = ({ datasetName, metadata }: ExportJobDetailsProps) => {
    const { t } = useTranslation();
    const { data: selectedProject } = useProject();

    const projectLabels = selectedProject.task.labels ?? [];
    const exportLabelsNames = metadata.filters.labels ?? [];

    const projectLabelsNames = projectLabels.map((label) => label.name);
    const selectedLabels = exportLabelsNames.filter((name) => projectLabelsNames.includes(name));

    const labelsList = isEmpty(selectedLabels) ? projectLabelsNames : selectedLabels;

    return (
        <Flex direction={'column'}>
            <Text UNSAFE_style={{ fontWeight: 500, fontSize: dimensionValue('size-225') }}>
                {t('dataset.export.details.heading', {
                    name: isNil(datasetName) ? t('dataset.export.details.defaultName') : datasetName,
                })}
            </Text>

            <Grid
                marginTop={'size-200'}
                alignItems={'center'}
                gap='size-125'
                columns={['auto', '1px', 'auto', '1px', '1fr']}
            >
                <Text>
                    {t('dataset.export.details.format')}{' '}
                    <Text
                        UNSAFE_style={{
                            textTransform: isGetiFormat(metadata.export_format) ? 'capitalize' : 'uppercase',
                        }}
                    >
                        {metadata.export_format}
                    </Text>{' '}
                </Text>

                <Divider orientation='vertical' size='S' />

                <Text>
                    {t('dataset.export.details.media')}{' '}
                    {metadata.filters.include_unannotated
                        ? t('dataset.export.details.allMedia')
                        : t('dataset.export.details.onlyAnnotated')}
                </Text>

                <Divider orientation='vertical' size='S' />

                <Text UNSAFE_style={{ overflow: 'hidden', whiteSpace: 'nowrap', textOverflow: 'ellipsis' }}>
                    {t('dataset.export.details.labels')} {labelsList.join(', ')}
                </Text>
            </Grid>
        </Flex>
    );
};
