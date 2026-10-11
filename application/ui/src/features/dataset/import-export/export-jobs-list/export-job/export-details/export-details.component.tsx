// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { ExportDatasetMetadata } from '@/api/types';
import { useTranslation } from '@/i18n';
import { dimensionValue, Divider, Flex, Grid, Text } from '@geti-ui/ui';
import { useOptionalDatasetViewsQuery } from 'hooks/api/dataset-views.hook';
import { isEmpty, isNil } from 'lodash-es';

import { useProject } from '../../../../../../hooks/api/project.hook';
import { isNonEmptyString } from '../../../../../../shared/util';

type ExportJobDetailsProps = {
    datasetName?: string;
    metadata: ExportDatasetMetadata;
};

const isGetiFormat = (format?: string | null) => format?.toLowerCase() === 'geti';

const useDatasetViewName = (datasetViewId: string | null | undefined) => {
    const { t } = useTranslation();
    const { data: datasetViews } = useOptionalDatasetViewsQuery(isNonEmptyString(datasetViewId));

    if (!isNonEmptyString(datasetViewId)) {
        return undefined;
    }

    return datasetViews?.find(({ id }) => id === datasetViewId)?.name ?? t('dataset.views.deleted');
};

export const ExportJobDetails = ({ datasetName, metadata }: ExportJobDetailsProps) => {
    const { t } = useTranslation();
    const { data: selectedProject } = useProject();
    const datasetViewName = useDatasetViewName(metadata.dataset_view_id);

    const projectLabels = selectedProject.task.labels ?? [];
    const exportLabelsNames = metadata.filters.labels ?? [];

    const projectLabelsNames = projectLabels.map((label) => label.name);
    const selectedLabels = exportLabelsNames.filter((name) => projectLabelsNames.includes(name));

    const labelsList = isEmpty(selectedLabels) ? projectLabelsNames : selectedLabels;

    const title = isNil(metadata.dataset_view_id)
        ? t('dataset.export.heading', { name: datasetName ?? t('dataset.export.details.defaultName') })
        : t('dataset.export.headingDatasetView');

    return (
        <Flex direction={'column'}>
            <Text UNSAFE_style={{ fontWeight: 500, fontSize: dimensionValue('size-225') }}>{title}</Text>

            <Grid
                marginTop={'size-200'}
                alignItems={'center'}
                gap='size-125'
                columns={
                    datasetViewName === undefined
                        ? ['auto', '1px', 'auto', '1px', '1fr']
                        : ['auto', '1px', 'auto', '1px', 'auto', '1px', '1fr']
                }
            >
                {isNonEmptyString(datasetViewName) && (
                    <>
                        <Text>{t('dataset.export.datasetView', { name: datasetViewName })}</Text>

                        <Divider orientation='vertical' size='S' />
                    </>
                )}

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
                        ? t('dataset.filters.allMedia')
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
