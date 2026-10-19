// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode, Suspense, useState } from 'react';

import type { TaskType } from '@/api/types';
import { GetiLogoLink } from '@/components/geti-logo-link/geti-logo-link.component';
import { HelpMenu } from '@/components/help-menu/help-menu.component';
import { LanguagesSelector } from '@/components/languages-selector/languages-selector.component';
import { useTranslation } from '@/i18n';
import { Content, Divider, Flex, Grid, Loading, Text, View } from '@geti-ui/ui';
import { useProjects } from 'hooks/api/project.hook';
import { partition } from 'lodash-es';

import { version } from '../../../../package.json';
import { isNonEmptyArray } from '../../../shared/util';
import { EmptyProjectList } from './empty-project-list/empty-project-list.component';
import { NoMatchingProjects } from './filter-projects/no-matching-projects.component';
import { ProjectFilters } from './filter-projects/project-filters.component';
import { useProjectFilters } from './filter-projects/use-project-filters.hook';
import { ImportJobsList } from './import-jobs-list/import-jobs-list.component';
import { NewProjectCard } from './new-project-card/new-project-card.component';
import { ProjectCard } from './project-card.component';
import { SORT_BY_HANDLERS, SortProjects } from './sort-projects/sort-projects.component';
import { SortBy } from './sort-projects/utils';

import classes from './project-list.module.scss';

const CARD_WIDTH = 280;

const ProjectSidebar = ({
    shouldShowFilters,
    searchName,
    setSearchName,
    selectedTaskTypes,
    setSelectedTaskTypes,
    onlyWithTrainedModels,
    setOnlyWithTrainedModels,
}: {
    shouldShowFilters: boolean;
    searchName: string;
    setSearchName: (value: string) => void;
    selectedTaskTypes: TaskType[];
    setSelectedTaskTypes: (taskTypes: TaskType[]) => void;
    onlyWithTrainedModels: boolean;
    setOnlyWithTrainedModels: (value: boolean) => void;
}) => {
    return (
        <Flex direction={'column'} gap={'size-300'} justifyContent={'space-between'} UNSAFE_className={classes.sidebar}>
            <Flex direction={'column'} gap={'size-300'}>
                <GetiLogoLink />

                <Divider size={'S'} />

                <NewProjectCard />

                {shouldShowFilters && (
                    <>
                        <Divider size={'S'} />

                        <ProjectFilters
                            searchName={searchName}
                            onSearchChange={setSearchName}
                            selectedTaskTypes={selectedTaskTypes}
                            onSelectedTaskTypesChange={setSelectedTaskTypes}
                            onlyWithTrainedModels={onlyWithTrainedModels}
                            onOnlyWithTrainedModelsChange={setOnlyWithTrainedModels}
                        />
                    </>
                )}
            </Flex>

            <Text UNSAFE_className={classes.version}>Geti v{version}</Text>
        </Flex>
    );
};

const ProjectListActions = ({ jobsButton }: { jobsButton?: ReactNode }) => {
    return (
        <>
            {jobsButton}
            <LanguagesSelector />
            <HelpMenu />
        </>
    );
};

const ProjectGrid = ({ jobsButton }: { jobsButton?: ReactNode }) => {
    const { t } = useTranslation();
    const projectsQuery = useProjects();
    const projects = projectsQuery.data;
    const [sortBy, setSortBy] = useState<SortBy>('createdAt-descending');
    const hasProjects = isNonEmptyArray(projects);

    const [[activeProject], projectsWithoutActivePipeline] = partition(projects, (project) => project.active_pipeline);

    const shouldShowFilters = projectsWithoutActivePipeline.length > 0;

    const {
        searchName,
        setSearchName,
        selectedTaskTypes,
        setSelectedTaskTypes,
        onlyWithTrainedModels,
        setOnlyWithTrainedModels,
        filteredProjects,
        isFiltering,
    } = useProjectFilters(projectsWithoutActivePipeline);

    const sortedProjects = SORT_BY_HANDLERS[sortBy](filteredProjects);

    const projectNames = projects.map((project) => project.name);

    if (!hasProjects) {
        return (
            <Flex direction={'column'} height={'100%'}>
                <Flex
                    justifyContent={'end'}
                    alignItems={'center'}
                    gap={'size-75'}
                    UNSAFE_className={classes.emptyToolbar}
                >
                    <ProjectListActions />
                </Flex>
                <View flex={1} minHeight={0}>
                    <EmptyProjectList />
                </View>
            </Flex>
        );
    }

    const totalCount = projectsWithoutActivePipeline.length;
    const countLabel = isFiltering
        ? t('project.list.countFiltered', { count: totalCount, filtered: sortedProjects.length })
        : t('project.list.count', { count: totalCount });
    const visibleCardsCount = (activeProject === undefined ? 0 : 1) + sortedProjects.length;

    return (
        <Flex height={'100%'}>
            <ProjectSidebar
                shouldShowFilters={shouldShowFilters}
                searchName={searchName}
                setSearchName={setSearchName}
                selectedTaskTypes={selectedTaskTypes}
                setSelectedTaskTypes={setSelectedTaskTypes}
                onlyWithTrainedModels={onlyWithTrainedModels}
                setOnlyWithTrainedModels={setOnlyWithTrainedModels}
            />

            <Flex direction={'column'} gap={'size-300'} flex={1} minWidth={0} UNSAFE_className={classes.gridPanel}>
                <Flex width={'100%'} alignItems={'center'} justifyContent={'space-between'} gap={'size-300'}>
                    {shouldShowFilters && <SortProjects sortBy={sortBy} onSort={setSortBy} />}

                    <Flex marginStart={'auto'} alignItems={'center'} gap={'size-75'}>
                        {shouldShowFilters && <Text UNSAFE_className={classes.projectMetadata}>{countLabel}</Text>}
                        <ProjectListActions jobsButton={jobsButton} />
                    </Flex>
                </Flex>

                {isFiltering && sortedProjects.length === 0 && <NoMatchingProjects />}

                {visibleCardsCount > 0 && (
                    <Grid
                        flex={1}
                        gap={'size-200'}
                        autoRows={'min-content'}
                        columns={`repeat(auto-fill, minmax(${CARD_WIDTH}px, 1fr))`}
                        UNSAFE_className={classes.projectGrid}
                    >
                        {activeProject !== undefined && (
                            <ProjectCard
                                item={activeProject}
                                prioritizeImage
                                projectNames={projectNames.filter((projectName) => projectName !== activeProject.name)}
                            />
                        )}
                        {sortedProjects.map((item, index) => (
                            <ProjectCard
                                key={item.id}
                                item={item}
                                prioritizeImage={index === 0}
                                projectNames={projectNames.filter((projectName) => projectName !== item.name)}
                            />
                        ))}
                    </Grid>
                )}
            </Flex>
        </Flex>
    );
};

export const ProjectList = ({ jobsButton }: { jobsButton?: ReactNode }) => {
    return (
        <View height={'100%'} position={'relative'}>
            <Content height={'100%'} margin={'0'}>
                <Flex direction={'column'} height={'100%'}>
                    <ImportJobsList />

                    <View flex={1} UNSAFE_style={{ overflow: 'auto' }}>
                        <Suspense fallback={<Loading size='M' mode='inline' />}>
                            <ProjectGrid jobsButton={jobsButton} />
                        </Suspense>
                    </View>
                </Flex>
            </Content>
        </View>
    );
};
