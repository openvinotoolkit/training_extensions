// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Dispatch, SetStateAction } from 'react';

import type { TaskType } from '@/api/types';
import { useTranslation, type TranslateFn } from '@/i18n';
import { Divider, Flex, Grid, Heading, Image, Radio, RadioGroup, Text, View } from '@geti-ui/ui';

import classificationImageUrl from '../../../assets/classification.webp';
import detectionImageUrl from '../../../assets/detection.webp';
import segmentationImageUrl from '../../../assets/segmentation.webp';
import type { TaskOption } from './interface';

import classes from './task-selection.module.scss';

export const MAP_TASK_TYPE_TO_VERB_KEY = {
    detection: 'project.create.tasks.detection.verb',
    instance_segmentation: 'project.create.tasks.instanceSegmentation.verb',
    classification: 'project.create.tasks.classification.verb',
} as const satisfies Record<TaskType, string>;

const getTaskOptions = (t: TranslateFn): TaskOption[] => [
    {
        id: 'detection_task',
        imageSrc: detectionImageUrl,
        title: t('project.create.tasks.detection.title'),
        description: t('project.create.tasks.detection.description'),
        advice: t('project.create.tasks.detection.advice'),
        verb: t('project.create.tasks.detection.verb'),
        value: 'detection',
    },
    {
        id: 'segmentation_task',
        imageSrc: segmentationImageUrl,
        title: t('project.create.tasks.instanceSegmentation.title'),
        description: t('project.create.tasks.instanceSegmentation.description'),
        advice: t('project.create.tasks.instanceSegmentation.advice'),
        verb: t('project.create.tasks.instanceSegmentation.verb'),
        value: 'instance_segmentation',
    },
    {
        id: 'classification_task',
        imageSrc: classificationImageUrl,
        title: t('project.create.tasks.classification.title'),
        description: t('project.create.tasks.classification.description'),
        advice: t('project.create.tasks.classification.advice'),
        verb: t('project.create.tasks.classification.verb'),
        value: 'classification',
    },
];

type TaskOptionProps = {
    taskOption: TaskOption;
    onPress: () => void;
};

const Option = ({ taskOption, onPress }: TaskOptionProps) => {
    return (
        <div onClick={onPress} className={classes.option} aria-label={`Task option: ${taskOption.title}`}>
            <View>
                <Image height={'size-2400'} width={'100%'} src={taskOption.imageSrc} alt={taskOption.title} />
            </View>

            <View padding={'size-200'}>
                <Flex justifyContent={'space-between'} gap={'size-50'} alignItems={'center'}>
                    <Heading level={2} UNSAFE_className={classes.title}>
                        {taskOption.title}
                    </Heading>
                    <Radio aria-label={taskOption.value} value={taskOption.value} />
                </Flex>

                <Text UNSAFE_className={classes.description}>{taskOption.description}</Text>

                <Divider marginTop={'size-100'} marginBottom={'size-150'} size={'S'} />

                <Text>{taskOption.advice}</Text>
            </View>
        </div>
    );
};

type TaskSelectionProps = { selectedTask: TaskType | null; setSelectedTask: Dispatch<SetStateAction<TaskType | null>> };

export const TaskSelection = ({ selectedTask, setSelectedTask }: TaskSelectionProps) => {
    const { t } = useTranslation();
    const taskOptions = getTaskOptions(t);
    const selectedTaskOption = taskOptions.find((task) => task.value === selectedTask);

    return (
        <Flex direction={'column'} gap={'size-300'} alignItems={'center'}>
            <RadioGroup
                aria-label='Task selection'
                width={'100%'}
                value={selectedTaskOption?.value}
                onChange={(value: string) => {
                    const option = taskOptions.find((taskOption) => taskOption.value === value);

                    if (option) setSelectedTask(option.value);
                }}
            >
                <Grid
                    columns={
                        'repeat(3, minmax(min(100%, var(--spectrum-global-dimension-size-3600)), ' +
                        'var(--spectrum-global-dimension-size-4600)))'
                    }
                    gap={'size-300'}
                    width={'100%'}
                    justifyContent={'center'}
                >
                    {taskOptions.map((taskOption) => (
                        <Option
                            key={taskOption.value}
                            taskOption={taskOption}
                            onPress={() => {
                                setSelectedTask(taskOption.value);
                            }}
                        />
                    ))}
                </Grid>
            </RadioGroup>
        </Flex>
    );
};
