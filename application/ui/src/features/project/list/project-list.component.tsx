// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Suspense, useMemo, useState } from 'react';

import { Content, Divider, Flex, Grid, Heading, Loading, Text, View } from '@geti-ui/ui';
import { useProjects } from 'hooks/api/project.hook';
import { partition } from 'lodash-es';
import { useTranslation } from 'react-i18next';

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

import backgroundStyles from '../project-background.module.scss';
import classes from './project-list.module.scss';

const ProjectGrid = () => {
    const { t } = useTranslation();
    const projectsQuery = useProjects();
    const projects = projectsQuery.data;
    const [sortBy, setSortBy] = useState<SortBy>('createdAt-descending');
    const hasProjects = isNonEmptyArray(projects);

    const [[activeProject], projectsWithoutActivePipeline] = partition(projects, (project) => project.active_pipeline);

    const shouldShowFilters = projectsWithoutActivePipeline.length > 0;

    const { searchName, setSearchName, selectedTaskTypes, setSelectedTaskTypes, filteredProjects, isFiltering } =
        useProjectFilters(projectsWithoutActivePipeline);

    const sortedProjects = useMemo(() => {
        return SORT_BY_HANDLERS[sortBy](filteredProjects);
    }, [filteredProjects, sortBy]);

    const projectNames = projects.map((project) => project.name);

    if (!hasProjects) {
        return <EmptyProjectList />;
    }

    const totalCount = projectsWithoutActivePipeline.length;
    const countLabel = isFiltering
        ? t('project.list.countFiltered', { count: totalCount, filtered: sortedProjects.length })
        : t('project.list.count', { count: totalCount });
    const cardMinWidth = 480;
    const visibleCardsCount = (activeProject === undefined ? 0 : 1) + sortedProjects.length;

    return (
        <Flex direction={'column'} gap={'size-300'} height={'100%'}>
            <View UNSAFE_className={classes.newProjectRow}>
                <NewProjectCard />
            </View>

            <Divider size={'S'} />

            <Heading width={'100%'} level={1} UNSAFE_className={classes.heading}>
                {t('project.list.heading')}
            </Heading>

            {shouldShowFilters && (
                <Flex width={'100%'} gap={'size-200'}>
                    <SortProjects sortBy={sortBy} onSort={setSortBy} />

                    <Divider size={'S'} orientation={'vertical'} />

                    <Flex flex={1} alignItems={'center'} gap={'size-200'}>
                        <Text UNSAFE_className={classes.projectMetadata}>{countLabel}</Text>

                        <ProjectFilters
                            searchName={searchName}
                            onSearchChange={setSearchName}
                            selectedTaskTypes={selectedTaskTypes}
                            onSelectedTaskTypesChange={setSelectedTaskTypes}
                        />
                    </Flex>
                </Flex>
            )}
            {isFiltering && sortedProjects.length === 0 && <NoMatchingProjects />}

            {visibleCardsCount > 0 && (
                <Grid
                    flex={1}
                    gap={'size-300'}
                    autoRows={'size-2000'}
                    columns={
                        visibleCardsCount === 1
                            ? `minmax(min(${cardMinWidth}px, 100%), .5fr)`
                            : `repeat(auto-fit, minmax(min(${cardMinWidth}px, 100%), 1fr))`
                    }
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
    );
};

const AppInfo = () => {
    return <Text UNSAFE_className={classes.version}>v{version}</Text>;
};

export const ProjectList = () => {
    return (
        <View UNSAFE_className={backgroundStyles.projectBackground} height={'100%'} position={'relative'}>
            <Content height={'100%'} maxWidth={'1560px'} margin={'0 auto'} UNSAFE_className={classes.content}>
                <Flex direction={'column'} height={'100%'}>
                    <ImportJobsList />

                    <View flex={1} UNSAFE_style={{ overflow: 'auto' }}>
                        <Suspense fallback={<Loading size='M' mode='inline' />}>
                            <ProjectGrid />
                        </Suspense>
                    </View>
                </Flex>

                <View bottom={'size-150'} left={'size-150'} position={'absolute'}>
                    <AppInfo />
                </View>
            </Content>
        </View>
    );
};
