// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { TaskType } from '@/api/types';
import { FilterPopoverButton } from '@/components/filter-popover-button/filter-popover-button.component';
import { useTranslation } from '@/i18n';
import { Checkbox, CheckboxGroup, Flex, SearchField, View } from '@geti-ui/ui';
import { isEmpty } from 'lodash-es';

import { MAP_PROJECT_TYPE_TO_TITLE_KEY } from '../util';
import { TASK_TYPE_OPTIONS } from './utils';

type ProjectFiltersProps = {
    searchName: string;
    onSearchChange: (value: string) => void;
    selectedTaskTypes: TaskType[];
    onSelectedTaskTypesChange: (taskTypes: TaskType[]) => void;
};

export const ProjectFilters = ({
    searchName,
    onSearchChange,
    selectedTaskTypes,
    onSelectedTaskTypesChange,
}: ProjectFiltersProps) => {
    const { t } = useTranslation();

    const summary = isEmpty(selectedTaskTypes)
        ? null
        : t('project.list.filters.typesSelected', { count: selectedTaskTypes.length });

    return (
        <Flex alignItems={'center'} gap={'size-200'} flex={1}>
            <SearchField
                value={searchName}
                onChange={onSearchChange}
                placeholder={t('project.list.filters.searchPlaceholder')}
                aria-label={t('project.list.filters.searchLabel')}
                flex={1}
            />

            <View backgroundColor={'gray-50'}>
                <FilterPopoverButton
                    ariaLabel={t('project.list.filters.taskType')}
                    placeholder={t('project.list.filters.taskType')}
                    summary={summary}
                    minWidth={'size-2400'}
                    dialogWidth={'size-1600'}
                >
                    <CheckboxGroup
                        aria-label={t('project.list.filters.taskType')}
                        value={selectedTaskTypes}
                        onChange={(values) => onSelectedTaskTypesChange(values as TaskType[])}
                    >
                        {TASK_TYPE_OPTIONS.map((taskType) => (
                            <Checkbox key={taskType} value={taskType}>
                                {t(MAP_PROJECT_TYPE_TO_TITLE_KEY[taskType])}
                            </Checkbox>
                        ))}
                    </CheckboxGroup>
                </FilterPopoverButton>
            </View>
        </Flex>
    );
};
