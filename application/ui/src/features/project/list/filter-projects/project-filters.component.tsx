// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { TaskType } from '@/api/types';
import { useTranslation } from '@/i18n';
import { Checkbox, CheckboxGroup, Flex, SearchField } from '@geti-ui/ui';

import { MAP_PROJECT_TYPE_TO_TITLE_KEY } from '../util';
import { TASK_TYPE_OPTIONS } from './utils';

import classes from './project-filters.module.scss';

const TRAINED_MODELS_OPTION = 'trained-models';

type ProjectFiltersProps = {
    searchName: string;
    onSearchChange: (value: string) => void;
    selectedTaskTypes: TaskType[];
    onSelectedTaskTypesChange: (taskTypes: TaskType[]) => void;
    onlyWithTrainedModels: boolean;
    onOnlyWithTrainedModelsChange: (value: boolean) => void;
};

export const ProjectFilters = ({
    searchName,
    onSearchChange,
    selectedTaskTypes,
    onSelectedTaskTypesChange,
    onlyWithTrainedModels,
    onOnlyWithTrainedModelsChange,
}: ProjectFiltersProps) => {
    const { t } = useTranslation();

    return (
        <Flex direction={'column'} gap={'size-300'}>
            <SearchField
                width={'100%'}
                value={searchName}
                onChange={onSearchChange}
                placeholder={t('project.list.filters.searchPlaceholder')}
                aria-label={'Search projects by name'}
            />

            <CheckboxGroup
                label={t('project.list.filters.taskType')}
                value={selectedTaskTypes}
                onChange={(values) => onSelectedTaskTypesChange(values as TaskType[])}
                UNSAFE_className={classes.filterGroup}
            >
                {TASK_TYPE_OPTIONS.map((taskType) => (
                    <Checkbox key={taskType} value={taskType}>
                        {t(MAP_PROJECT_TYPE_TO_TITLE_KEY[taskType])}
                    </Checkbox>
                ))}
            </CheckboxGroup>

            <CheckboxGroup
                label={t('project.list.filters.models')}
                value={onlyWithTrainedModels ? [TRAINED_MODELS_OPTION] : []}
                onChange={(values) => onOnlyWithTrainedModelsChange(values.includes(TRAINED_MODELS_OPTION))}
                UNSAFE_className={classes.filterGroup}
            >
                <Checkbox value={TRAINED_MODELS_OPTION}>{t('project.list.filters.withTrainedModels')}</Checkbox>
            </CheckboxGroup>
        </Flex>
    );
};
