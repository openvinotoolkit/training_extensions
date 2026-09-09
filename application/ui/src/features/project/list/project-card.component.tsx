// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import type { Project } from '@/api/types';
import { useTranslation } from '@/i18n';
import { Badge, dimensionValue, Flex, Heading, Text, View } from '@geti-ui/ui';
import { useQueryClient } from '@tanstack/react-query';
import { clsx } from 'clsx';
import { getProjectQueryOptions } from 'hooks/api/project.hook';
import { NavLink } from 'react-router-dom';

import placeholderThumbnailIconUrl from '../../../assets/icons/image-icon.svg?url';
import { paths } from '../../../constants/paths';
import { getProjectThumbnailUrl } from '../../../shared/media-url.utils';
import { ActiveProjectBadge } from './active-project-badge/active-project-badge.component';
import { MenuActions } from './menu-actions/menu-actions.component';
import { formatCreationDate, getProjectTypeTitle } from './util';

import classes from './project-list.module.scss';

const cardPadding = 'size-200';

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
    const isActive = item.active_pipeline;
    const taskType = getProjectTypeTitle(item.task, t);
    const queryClient = useQueryClient();

    const prefetchProject = () => {
        void queryClient.prefetchQuery(getProjectQueryOptions(item.id));
    };

    return (
        <div style={{ position: 'relative' }} aria-label={t('project.list.card.label', { projectName: item.name })}>
            <NavLink
                to={paths.project.dataset.index({ projectId: item.id })}
                viewTransition
                onPointerEnter={prefetchProject}
                onFocus={prefetchProject}
            >
                <Flex UNSAFE_className={clsx({ [classes.card]: true, [classes.activeCard]: isActive })}>
                    <View
                        height={'100%'}
                        backgroundColor={'gray-100'}
                        borderEndColor={'gray-75'}
                        borderEndWidth={'thick'}
                        width={'size-2000'}
                    >
                        <Flex height={'100%'} width={'100%'} alignItems={'center'} justifyContent={'center'}>
                            <ProjectThumbnail project={item} prioritizeImage={prioritizeImage} />
                        </Flex>
                    </View>

                    <View flex={1} padding={cardPadding}>
                        <Flex alignItems={'center'} justifyContent={'space-between'}>
                            <Heading level={2} marginEnd={'size-400'} UNSAFE_className={classes.projectCardName}>
                                <span title={item.name}>{item.name}</span>
                            </Heading>
                        </Flex>

                        <Flex gap={'size-50'}>
                            {taskType !== undefined && <ProjectTypeBadge type={taskType} />}
                            {isActive && <ActiveProjectBadge />}
                        </Flex>

                        <Flex marginTop={'size-200'} gap={'size-100'} direction={'column'}>
                            <Text UNSAFE_className={classes.projectMetadata}>
                                {t('project.list.card.created', { date: formatCreationDate(item.created_at) })}
                            </Text>
                            <Text UNSAFE_className={clsx(classes.labelList, classes.projectMetadata)}>
                                {t('project.list.card.labels', {
                                    labels: (item.task.labels ?? []).map((label) => label.name).join(', '),
                                })}
                            </Text>
                        </Flex>
                    </View>
                </Flex>
            </NavLink>

            <MenuActions
                projectId={item.id}
                projectName={item.name}
                projectNames={projectNames}
                isPipelineRunning={item.active_pipeline}
                actionButtonStyle={{
                    top: dimensionValue(cardPadding),
                    right: dimensionValue(cardPadding),
                    position: 'absolute',
                }}
            />
        </div>
    );
};
