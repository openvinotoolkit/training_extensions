// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useActionState } from 'react';

import { DatasetStatistics } from '@/components/dataset-statistics/dataset-statistics.component';
import { MultiSelectList } from '@/components/multi-select-list/multi-select-list.component';
import { Checkbox, dimensionValue, Divider, Flex, Form, Heading, View } from '@geti-ui/ui';
import { useSubmitJob } from 'hooks/api/jobs/jobs.hook';
import { useStagedDataset } from 'hooks/api/staged-dataset.hook';
import { useImportDatasetAsNewProject } from 'hooks/storage/use-import-dataset-as-new-project.hook';
import { isNil, isString } from 'lodash-es';

import { useImportDatasetDialog } from '../../../providers/import-dataset-dialog-provider.component';
import { LABEL_MAPPING_FORM_ID } from './util';

type LabelMappingProps = {
    stagedDatasetId: string;
};

type FormValues = { labels: string[]; include_unannotated: boolean };

const initialFormState: FormValues = {
    include_unannotated: true,
    labels: [],
};

const useFormConfig = (stagedDatasetId: string) => {
    const { datasetImportDialogState } = useImportDatasetDialog();
    const { getImportEntry, updateImportEntry } = useImportDatasetAsNewProject();
    const importEntry = getImportEntry(stagedDatasetId);
    const importDatasetJobMutation = useSubmitJob();

    return useActionState<FormValues, FormData>(async (_prevState, formData) => {
        const filters = {
            labels: formData.getAll('labels').filter(isString),
            include_unannotated: formData.get('include_unannotated') === 'on',
        };

        if (isNil(importEntry?.project)) {
            return filters;
        }

        const project = { ...importEntry.project, exclusive_labels: false };

        await importDatasetJobMutation.mutateAsync(
            {
                body: {
                    job_type: 'import_dataset_as_new_project',
                    staged_dataset_id: stagedDatasetId,
                    parameters: { project, filters },
                },
            },
            {
                onSuccess: ({ job_id }) => {
                    updateImportEntry(stagedDatasetId, { filters, project, importJobId: job_id, step: 'importing' });
                },
                onSettled: datasetImportDialogState.close,
            }
        );

        return filters;
    }, initialFormState);
};

export const ImportLabelMapping = ({ stagedDatasetId }: LabelMappingProps) => {
    const { data: stagedDataset } = useStagedDataset(stagedDatasetId);

    const [formState, submitAction] = useFormConfig(stagedDatasetId);

    const datasetLabels = stagedDataset?.metadata?.labels ?? [];

    const totalImages = stagedDataset?.metadata?.num_images ?? 0;
    const totalAnnotatedImages = stagedDataset?.metadata?.num_annotated_images ?? 0;

    const totalFrames = stagedDataset?.metadata?.num_frames ?? 0;
    const totalAnnotatedFrames = stagedDataset?.metadata?.num_annotated_frames ?? 0;

    return (
        <Flex direction={'column'} gap={'size-200'} UNSAFE_style={{ padding: dimensionValue('size-275') }}>
            <Heading>Imported dataset statistics</Heading>

            <View padding={'size-200'} borderRadius={'regular'} backgroundColor={'gray-75'}>
                <Flex justifyContent={'center'} gap={'size-200'}>
                    <DatasetStatistics
                        label='images'
                        totalMediaItems={totalImages}
                        totalAnnotatedItems={totalAnnotatedImages}
                    />

                    {totalFrames > 0 && (
                        <DatasetStatistics
                            label='frames'
                            totalMediaItems={totalFrames}
                            totalAnnotatedItems={totalAnnotatedFrames}
                        />
                    )}
                </Flex>
            </View>

            <Heading marginTop={'size-200'}>Label mapping</Heading>

            <View padding={'size-200'} borderRadius={'regular'} backgroundColor={'gray-75'}>
                <Form id={LABEL_MAPPING_FORM_ID} validationBehavior='native' action={submitAction}>
                    <MultiSelectList
                        name='labels'
                        label='Dataset labels'
                        maxHeight='size-2000'
                        defaultSelectedKeys={new Set(datasetLabels.map((label) => label))}
                        items={datasetLabels.map((label) => ({ id: label, name: label }))}
                    />

                    <Divider size='S' marginTop={'size-200'} />

                    <Checkbox
                        defaultSelected={formState.include_unannotated}
                        name='include_unannotated'
                        aria-label='include unannotated'
                    >
                        Include media without annotations
                    </Checkbox>
                </Form>
            </View>
        </Flex>
    );
};
