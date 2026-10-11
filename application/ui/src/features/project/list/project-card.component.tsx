// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import type { Project } from '@/api/types';
import placeholderThumbnailIconUrl from '@/assets/icons/image-icon.svg?url';
import { useTranslation } from '@/i18n';
import { Badge, Flex, Heading, Text, View } from '@geti-ui/ui';
import { useQueryClient } from '@tanstack/react-query';
import { clsx } from 'clsx';
import { getProjectQueryOptions } from 'hooks/api/project.hook';
import { LONG_DATE_TIME_FORMAT, useFormatDate } from 'hooks/use-format-date.hook';
import { NavLink } from 'react-router';

import { paths } from '../../../constants/paths';
import { getProjectThumbnailUrl } from '../../../shared/media-url.utils';
import { ActiveProjectBadge } from './active-project-badge/active-project-badge.component';
import { MenuActions } from './menu-actions/menu-actions.component';
import { ProjectLabelsButton } from './project-labels-button/project-labels-button.component';
import { getProjectTypeTitle } from './util';

import classes from './project-list.module.scss';

type ProjectTypeBadgeProps = {
    type: string;
};

const ProjectTypeBadge = ({ type }: ProjectTypeBadgeProps) => {
    return (
        <Badge variant={'neutral'} UNSAFE_className={classes.tag}>
            <Text>{type}</Text>
        </Badge>
    );
};

type ProjectThumbnailProps = {
    project: Project;
    prioritizeImage?: boolean;
};

const ProjectThumbnail = ({ project, prioritizeImage }: ProjectThumbnailProps) => {
    const [isThumbnailLoadingError, setIsThumbnailLoadingError] = useState<boolean>(false);

    const src = isThumbnailLoadingError ? placeholderThumbnailIconUrl : getProjectThumbnailUrl(project.id);

    return (
        <img
            src={src}
            alt={project.name}
            loading={prioritizeImage ? 'eager' : 'lazy'}
            fetchPriority={prioritizeImage ? 'high' : 'auto'}
            onError={() => setIsThumbnailLoadingError(true)}
            className={clsx(classes.thumbnail, { [classes.thumbnailError]: isThumbnailLoadingError })}
        />
    );
};

type ProjectCardProps = {
    item: Project;
    prioritizeImage?: boolean;
    projectNames: string[];
};

export const ProjectCard = ({ item, prioritizeImage = false, projectNames }: ProjectCardProps) => {
    const { t } = useTranslation();
    const formatDate = useFormatDate(LONG_DATE_TIME_FORMAT);
    const isActive = item.active_pipeline;
    const taskType = getProjectTypeTitle(item.task, t);
    const labels = item.task.labels ?? [];
    const queryClient = useQueryClient();

    const prefetchProject = () => {
        void queryClient.prefetchQuery(getProjectQueryOptions(item.id));
    };

    return (
        <div className={clsx(classes.card, { [classes.activeCard]: isActive })} aria-label={`Project: ${item.name}`}>
            <NavLink
                to={paths.project.dataset.index({ projectId: item.id })}
                viewTransition
                onPointerEnter={prefetchProject}
                onFocus={prefetchProject}
            >
                <Flex direction={'column'} height={'100%'}>
                    <View position={'relative'} UNSAFE_className={classes.thumbnailContainer}>
                        <ProjectThumbnail project={item} prioritizeImage={prioritizeImage} />

                        <Flex gap={'size-50'} UNSAFE_className={classes.cardBadges}>
                            {taskType !== undefined && <ProjectTypeBadge type={taskType} />}
                            {isActive && <ActiveProjectBadge />}
                        </Flex>
                    </View>

                    <View flex={1} padding={'size-100'}>
                        <Flex direction={'column'} gap={'size-75'}>
                            <Heading
                                level={3}
                                marginEnd={'size-900'}
                                marginY={0}
                                UNSAFE_className={classes.projectCardName}
                            >
                                <span title={item.name}>{item.name}</span>
                            </Heading>

                            <Text UNSAFE_className={classes.projectMetadata}>
                                {t('project.list.card.created', { date: formatDate(item.created_at) ?? '-' })}
                            </Text>
                        </Flex>
                    </View>
                </Flex>
            </NavLink>

            <Flex gap={'size-50'} UNSAFE_className={classes.cardActions}>
                <ProjectLabelsButton labels={labels} />

                <MenuActions
                    projectId={item.id}
                    projectName={item.name}
                    projectNames={projectNames}
                    isPipelineRunning={item.active_pipeline}
                />
            </Flex>
        </div>
    );
};
