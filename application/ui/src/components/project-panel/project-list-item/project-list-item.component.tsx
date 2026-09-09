// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Project } from '@/api/types';
import { useTranslation } from '@/i18n';
import { Badge, Flex, Text } from '@geti-ui/ui';
import { useNavigate } from 'react-router-dom';

import { paths } from '../../../constants/paths';
import { ActiveProjectBadge } from '../../../features/project/list/active-project-badge/active-project-badge.component';
import {
    ProjectActionsMenu,
    type ProjectActionMetadata,
} from '../../../features/project/list/menu-actions/menu-actions.component';
import { getProjectTypeTitle } from '../../../features/project/list/util';
import { ProjectThumbnail } from '../project-thumbnail/project-thumbnail.component';

import classes from './project-list-item.module.scss';

type ProjectListItemProps = {
    project: Project;
    projectNames: string[];
    onRename: (target: ProjectActionMetadata) => void;
    onDelete: (target: ProjectActionMetadata) => void;
    onEnableBlocked: (target: ProjectActionMetadata) => void;
};

export const ProjectListItem = ({
    project,
    projectNames,
    onRename,
    onDelete,
    onEnableBlocked,
}: ProjectListItemProps) => {
    const { t } = useTranslation();
    const navigate = useNavigate();

    const taskType = getProjectTypeTitle(project.task, t);

    const handleNavigateToProject = () => {
        navigate(paths.project.dataset.index({ projectId: project.id }), {
            viewTransition: true,
        });
    };

    return (
        <li className={classes.projectListItem} onClick={handleNavigateToProject}>
            <Flex justifyContent='space-between' alignItems='center' marginX={'size-200'}>
                <Flex alignItems={'center'} gap={'size-100'} minWidth={0}>
                    <ProjectThumbnail project={project} height={'size-300'} width={'size-300'} />
                    <Text UNSAFE_className={classes.projectListItemName}>
                        <span title={project.name}>{project.name}</span>
                    </Text>
                    {taskType !== undefined && (
                        <Badge variant={'neutral'} UNSAFE_className={classes.itemTag}>
                            <Text>{taskType}</Text>
                        </Badge>
                    )}
                    {project.active_pipeline && <ActiveProjectBadge size='S' />}
                </Flex>

                <ProjectActionsMenu
                    projectId={project.id}
                    projectName={project.name}
                    isPipelineRunning={project.active_pipeline}
                    projectNames={projectNames}
                    onRename={onRename}
                    onDelete={onDelete}
                    onEnableBlocked={onEnableBlocked}
                />
            </Flex>
        </li>
    );
};
