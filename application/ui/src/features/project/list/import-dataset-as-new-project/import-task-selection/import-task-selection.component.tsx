// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useActionState, useState } from 'react';

import type { TaskType } from '@/api/types';
import { useTranslation } from '@/i18n';
import { Flex, Form, Item, Picker, Text, TextField, View } from '@geti-ui/ui';
import { InfoOutline } from '@geti-ui/ui/icons';
import { useProjects } from 'hooks/api/project.hook';
import { useStagedDatasetSuspense } from 'hooks/api/staged-dataset.hook';
import { useImportDatasetAsNewProject } from 'hooks/storage/use-import-dataset-as-new-project.hook';

import { generateUniqueProjectName } from '../../../create/utils';
import { useImportDatasetDialog } from '../../../providers/import-dataset-dialog-provider.component';
import { validateProjectName } from '../../../validator';
import { MAP_PROJECT_TYPE_TO_TITLE_KEY } from '../../util';
import { getAllowedTaskTypes, getRecommendedTaskType, TASK_SELECTION_FORM_ID } from './util';

type ImportTaskSelectionProps = {
    stagedDatasetId: string;
};

const useFormConfig = (
    stagedDatasetId: string,
    defaultTaskType: TaskType | undefined,
    allowedTaskTypes: TaskType[]
) => {
    const { t } = useTranslation();
    const { data: projects } = useProjects();
    const { setCurrentStep } = useImportDatasetDialog();
    const { getImportEntry, updateImportEntry } = useImportDatasetAsNewProject();
    const importEntry = getImportEntry(stagedDatasetId);

    const uniqueProjectName = generateUniqueProjectName(
        projects.map((project) => project.name),
        t
    );

    const taskType = importEntry?.project?.task_type;
    const finalTaskType = taskType && allowedTaskTypes.includes(taskType) ? taskType : defaultTaskType;

    const initialFormState = {
        name: importEntry?.project?.name ?? uniqueProjectName,
        task_type: finalTaskType,
    };

    return useActionState<{ name: string; task_type: TaskType | undefined }, FormData>(async (_prevState, formData) => {
        const project = {
            name: String(formData.get('name')).trim(),
            task_type: formData.get('task_type') as TaskType,
        };

        setCurrentStep('labelMapping');
        updateImportEntry(stagedDatasetId, { project, step: 'labelMapping' });
        return project;
    }, initialFormState);
};

export const ImportTaskSelection = ({ stagedDatasetId }: ImportTaskSelectionProps) => {
    const { t } = useTranslation();
    const { data: projects } = useProjects();
    const { data: stagedDataset } = useStagedDatasetSuspense(stagedDatasetId);

    const annotationType = stagedDataset?.metadata?.annotation_type;
    const isGetiFormat = stagedDataset.format === 'geti';
    const allowedTaskTypes = getAllowedTaskTypes(annotationType);
    const defaultTaskType = isGetiFormat ? getRecommendedTaskType(annotationType) : undefined;

    const [formState, submitAction] = useFormConfig(stagedDatasetId, defaultTaskType, allowedTaskTypes);
    const [name, setName] = useState(formState.name);

    const validationErrorMessage = validateProjectName(
        name.trim(),
        projects.map((project) => project.name),
        t
    );

    const items = allowedTaskTypes.map((taskType) => ({
        key: taskType,
        label:
            defaultTaskType === taskType
                ? t('project.taskTypes.recommended', { taskType: t(MAP_PROJECT_TYPE_TO_TITLE_KEY[taskType]) })
                : t(MAP_PROJECT_TYPE_TO_TITLE_KEY[taskType]),
    }));

    return (
        <View backgroundColor={'gray-75'} margin={'size-300'} padding={'size-300'}>
            <Form id={TASK_SELECTION_FORM_ID} validationBehavior='native' action={submitAction}>
                <TextField
                    isRequired
                    name={'name'}
                    value={name}
                    onChange={setName}
                    label={t('project.import.taskSelection.projectName')}
                    aria-label={'Project name'}
                    defaultValue={formState.name}
                    marginBottom={'size-250'}
                    errorMessage={validationErrorMessage}
                    validationState={validationErrorMessage === undefined ? undefined : 'invalid'}
                />

                <Picker
                    isRequired
                    items={items}
                    name={'task_type'}
                    label={t('project.import.taskSelection.taskType')}
                    aria-label={'Task type'}
                    marginBottom={'size-150'}
                    placeholder={t('project.import.taskSelection.selectTaskPlaceholder')}
                    defaultSelectedKey={formState.task_type}
                >
                    {(item) => <Item>{item.label}</Item>}
                </Picker>

                <View>
                    {defaultTaskType !== undefined && (
                        <Flex gap='size-100' alignItems={'center'}>
                            <View width={16} height={16}>
                                <InfoOutline />
                            </View>

                            <Text>{t('project.import.taskSelection.recommendationHint')}</Text>
                        </Flex>
                    )}
                </View>
            </Form>
        </View>
    );
};
