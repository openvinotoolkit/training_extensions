// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { FormEvent, useState } from 'react';

import type { Label, Project, TaskType } from '@/api/types';
import { Button, ButtonGroup, Divider, Flex, Form, Text, TextField } from '@geti-ui/ui';
import { useCreateProject } from 'hooks/api/project.hook';
import { useTranslation } from 'react-i18next';
import { useNavigate } from 'react-router-dom';
import { v4 as uuid } from 'uuid';

import { paths } from '../../../constants/paths';
import { LabelSelection } from '../label-selection/label-selection.component';
import { MAP_TASK_TYPE_TO_VERB_KEY, TaskSelection } from '../task-selection/task-selection.component';
import { isClassificationTask } from '../task-type-guards';
import { PROJECT_NAME_MAX_LENGTH, validateProjectName } from '../validator';
import {
    ClassificationTaskSelection,
    ClassificationTaskType,
} from './classification-label-selection/classification-task-type-selection.component';
import { generateUniqueProjectName } from './utils';

import classes from './create-project-form.module.scss';

type CreateProjectFormProps = {
    projects: Project[];
};

export const CreateProjectForm = ({ projects }: CreateProjectFormProps) => {
    const { t } = useTranslation();
    const [selectedTask, setSelectedTask] = useState<TaskType | null>(null);
    const [labels, setLabels] = useState<Label[]>([]);
    const [name, setName] = useState<string>(() => generateUniqueProjectName(projects.map((project) => project.name)));
    const taskVerb = selectedTask === null ? undefined : t(MAP_TASK_TYPE_TO_VERB_KEY[selectedTask]);

    const [classificationTaskType, setClassificationTaskType] = useState<ClassificationTaskType>('single-label');

    const navigate = useNavigate();
    const createProjectMutation = useCreateProject();

    const isSubmitting = createProjectMutation.isPending || createProjectMutation.isSuccess;

    const validationErrorMessage = isSubmitting
        ? undefined
        : validateProjectName(
              name,
              projects.map((project) => project.name),
              t
          );

    const isSingleLabelClassification = isClassificationTask(selectedTask) && classificationTaskType === 'single-label';
    const needsMinimumNumberOfLabels = isSingleLabelClassification && labels.length < 2;

    const isCreateProjectDisabled =
        isSubmitting ||
        selectedTask === null ||
        validationErrorMessage !== undefined ||
        labels.length === 0 ||
        needsMinimumNumberOfLabels;

    const createProject = (e: FormEvent) => {
        e.preventDefault();

        if (isCreateProjectDisabled) {
            return;
        }

        const projectId = uuid();

        createProjectMutation.mutate(
            {
                body: {
                    id: projectId,
                    task: {
                        task_type: selectedTask,
                        exclusive_labels: isSingleLabelClassification,
                        labels,
                    },
                    name,
                },
            },
            {
                onSuccess: () => {
                    navigate(paths.project.dataset.index({ projectId }), {
                        viewTransition: true,
                    });
                },
            }
        );
    };

    return (
        <Form onSubmit={createProject} validationBehavior={'native'} height={'100%'}>
            <Flex
                flex={1}
                minHeight={0}
                width={'clamp(912px, 60vw, 1560px)'}
                margin={'0 auto'}
                gap={'size-500'}
                direction={'column'}
            >
                <Flex justifyContent={'center'} marginTop={'size-600'}>
                    <TextField
                        aria-label={t('project.create.nameFieldLabel')}
                        maxLength={PROJECT_NAME_MAX_LENGTH}
                        isRequired
                        value={name}
                        onChange={setName}
                        width={'50%'}
                        errorMessage={validationErrorMessage}
                        validationState={validationErrorMessage === undefined ? undefined : 'invalid'}
                    />
                </Flex>

                <Flex
                    direction='column'
                    gap='size-300'
                    UNSAFE_style={{ overflow: 'auto', margin: '0 auto' }}
                    width={'100%'}
                >
                    <Text UNSAFE_className={classes.taskTypeSelectionTitle}>{t('project.create.taskQuestion')}</Text>

                    <TaskSelection selectedTask={selectedTask} setSelectedTask={setSelectedTask} />

                    {isClassificationTask(selectedTask) && (
                        <ClassificationTaskSelection
                            selectedType={classificationTaskType}
                            onSelectedTypeChange={setClassificationTaskType}
                        />
                    )}

                    {selectedTask !== null && (
                        <Flex direction={'column'} alignItems={'center'} gap={'size-350'}>
                            <Flex>
                                <Text UNSAFE_className={classes.objectsToLearnTitle}>
                                    {t('project.create.labelsQuestion', { verb: taskVerb })}
                                </Text>
                            </Flex>
                            <LabelSelection labels={labels} setLabels={setLabels} taskType={selectedTask} />
                        </Flex>
                    )}
                </Flex>
            </Flex>

            <Flex direction={'column'} alignItems={'center'} UNSAFE_className={classes.buttonGroup} gap={'size-300'}>
                <Divider size={'S'} width={'100%'} />
                <ButtonGroup>
                    <Button
                        variant={'secondary'}
                        onPress={() => {
                            const isExternalReferrer =
                                document.referrer === '' ||
                                new URL(document.referrer).origin !== window.location.origin;

                            isExternalReferrer ? navigate(paths.project.index({})) : navigate(-1);
                        }}
                    >
                        {t('common.actions.goBack')}
                    </Button>
                    <Button type={'submit'} variant='accent' isDisabled={isCreateProjectDisabled}>
                        {t('project.create.submit')}
                    </Button>
                </ButtonGroup>
            </Flex>
        </Form>
    );
};
