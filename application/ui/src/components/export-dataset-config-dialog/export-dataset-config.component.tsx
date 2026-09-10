// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode, useState } from 'react';

import { useTranslation } from '@/i18n';
import {
    Button,
    ButtonGroup,
    Checkbox,
    Content,
    Dialog,
    DialogContainer,
    Divider,
    Flex,
    Form,
    Heading,
    Radio,
    RadioGroup,
    Text,
    View,
} from '@geti-ui/ui';
import { Alert, LinkOut } from '@geti-ui/ui/icons';
import { OverlayTriggerState } from '@react-stately/overlays';
import { useDatasetStatisticsQuery } from 'hooks/api/dataset.hook';
import { useProject } from 'hooks/api/project.hook';

import { useExportDatasetJobAction } from '../../hooks/use-export-dataset-job-action.hook';
import { Link } from '../../platform/components/link.component';
import { isEmptyLabel, useProjectLabelsWithEmptyLabel } from '../../shared/annotator/labels';
import { MultiSelectList } from '../multi-select-list/multi-select-list.component';
import { getFormatOptions } from '../util';

import classes from './export-dataset-config.module.scss';

const WarningMessages = ({ selectedExportFormat }: { selectedExportFormat: string | null }) => {
    const { t } = useTranslation();
    const isVisible = selectedExportFormat !== 'geti';

    const { data: statistics } = useDatasetStatisticsQuery(isVisible);
    const emptyLabel = useProjectLabelsWithEmptyLabel().find(isEmptyLabel);

    const isCocoFormatSelected = selectedExportFormat === 'coco';

    const hasVideos = (statistics?.media_counts.videos ?? 0) > 0;
    // Media annotated with the empty label is reported by the API under a `null` label id
    const emptyLabelName = statistics?.annotations_counts.instances_per_label.some(
        ({ label_id, instances }) => label_id === null && instances > 0
    )
        ? emptyLabel?.name
        : undefined;

    const unsupportedItems = [
        hasVideos && t('dataset.export.warnings.unsupportedItemsVideos'),
        emptyLabelName !== undefined && t('dataset.export.warnings.unsupportedItemsEmptyLabels'),
    ]
        .filter(Boolean)
        .join(' or ');

    if (!isVisible || (!hasVideos && emptyLabelName === undefined && !isCocoFormatSelected)) {
        return null;
    }

    return (
        <Flex alignItems={'start'} marginTop={'size-100'} gap={'size-100'}>
            <Flex>
                <Alert className={classes.warningMessageIcon} />
            </Flex>
            <Flex direction={'column'} gap={'size-75'}>
                {hasVideos && <Text>{t('dataset.export.warnings.videos')}</Text>}
                {emptyLabelName !== undefined && (
                    <Text>{t('dataset.export.warnings.emptyLabel', { emptyLabelName })}</Text>
                )}
                {unsupportedItems && (
                    <Text>{t('dataset.export.warnings.unsupportedItems', { items: unsupportedItems })}</Text>
                )}
                {isCocoFormatSelected && <Text>{t('dataset.export.warnings.subsetNotIncluded')}</Text>}
            </Flex>
        </Flex>
    );
};

type ExportDatasetConfigProps = {
    name?: string;
    datasetId: string | null;
    statistics: ReactNode;
    dialogState: OverlayTriggerState;
};

const FORM_ID = 'export-dataset-form';
const EXPORT_FORMATS_LINK =
    'https://docs.geti.intel.com/docs/user-guide/geti-fundamentals/datasets/dataset-export-import#supported-formats';

type ExportDatasetDialogContentProps = {
    name: string;
    datasetId: string | null;
    statistics: ReactNode;
    dialogState: OverlayTriggerState;
};

const ExportDatasetDialogContent = ({ name, datasetId, statistics, dialogState }: ExportDatasetDialogContentProps) => {
    const { t } = useTranslation();
    const { data: selectedProject } = useProject();

    const [formState, submitAction, isPending] = useExportDatasetJobAction({
        datasetId,
        onSuccess: dialogState.close,
    });

    const formatOptions = getFormatOptions(selectedProject.task.task_type);
    const [selectedExportFormat, setSelectedExportFormat] = useState<string | null>(formatOptions.at(0)?.value ?? null);

    const labels = selectedProject.task.labels?.map((label) => ({ id: label.name, name: label.name })) ?? [];

    return (
        <Dialog size='L' width={{ base: '70vw' }}>
            <Heading>{t('dataset.export.heading', { name })}</Heading>
            <Divider />
            <Content UNSAFE_className={classes.container}>
                <Heading>{t('dataset.export.statisticsHeading')}</Heading>
                {statistics}

                <Heading>{t('dataset.export.settingsHeading')}</Heading>

                <View backgroundColor='gray-75' padding='size-200' borderRadius='regular'>
                    <Form id={FORM_ID} validationBehavior='native' action={submitAction}>
                        <MultiSelectList
                            name='labels'
                            items={labels}
                            maxHeight='size-2000'
                            label={t('dataset.export.filterByLabel')}
                            ariaLabel='Filter annotations by label'
                            defaultSelectedKeys={new Set(labels.map(({ id }) => id))}
                        />

                        <Checkbox name='include_unannotated' defaultSelected={formState.include_unannotated}>
                            {t('dataset.export.includeUnannotated')}
                        </Checkbox>

                        <Divider size='S' />

                        <RadioGroup
                            name='export_format'
                            label={t('dataset.export.selectFormat')}
                            defaultValue={formState.export_format}
                            onChange={(value) => setSelectedExportFormat(value)}
                        >
                            {formatOptions.map((item) => (
                                <Radio key={item.value} value={item.value}>
                                    {item.label}
                                </Radio>
                            ))}
                        </RadioGroup>
                    </Form>

                    <WarningMessages selectedExportFormat={selectedExportFormat} />

                    <Link
                        href={EXPORT_FORMATS_LINK}
                        target='_blank'
                        rel='noopener noreferrer'
                        UNSAFE_className={classes.link}
                    >
                        {t('dataset.export.learnMoreFormats')}
                        <LinkOut size='XS' />
                    </Link>
                </View>
            </Content>

            <ButtonGroup>
                <Button onPress={dialogState.close} variant='secondary'>
                    {t('dataset.export.cancel')}
                </Button>
                <Button type='submit' form={FORM_ID} variant='accent' isPending={isPending} isDisabled={isPending}>
                    {t('dataset.export.submit')}
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};

export const ExportDatasetConfig = ({
    name = 'dataset',
    datasetId,
    statistics,
    dialogState,
}: ExportDatasetConfigProps) => {
    return (
        <DialogContainer onDismiss={dialogState.close}>
            {dialogState.isOpen && (
                <ExportDatasetDialogContent
                    name={name}
                    datasetId={datasetId}
                    statistics={statistics}
                    dialogState={dialogState}
                />
            )}
        </DialogContainer>
    );
};
