// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState, type CSSProperties } from 'react';

import { EnablePipelineBlockedDialog } from '@/components/enable-pipeline-blocked-dialog/enable-pipeline-blocked-dialog.component';
import { DeleteProjectDialog } from '@/components/project-dialogs/delete-project-dialog.component';
import { EditProjectNameDialog } from '@/components/project-dialogs/edit-project-name-dialog.component';
import { ActionButton, Item, Menu, MenuTrigger } from '@geti-ui/ui';
import { MoreMenu } from '@geti-ui/ui/icons';
import { useOverlayTriggerState } from '@react-stately/overlays';

import { useProjectMenuActions } from './use-project-menu-actions';

import classes from './menu-actions.module.scss';

type MenuActionsProps = {
    projectId: string;
    projectName: string;
    isPipelineRunning?: boolean;
    actionButtonStyle?: CSSProperties;
    projectNames: string[];
};

export type ProjectActionMetadata = {
    projectId: string;
    projectName: string;
    projectNames: string[];
};

type ProjectActionsMenuProps = MenuActionsProps & {
    onRename: (metadata: ProjectActionMetadata) => void;
    onDelete: (metadata: ProjectActionMetadata) => void;
    onEnableBlocked: (metadata: ProjectActionMetadata) => void;
};

export const ProjectActionsMenu = ({
    projectId,
    projectName,
    isPipelineRunning,
    actionButtonStyle,
    projectNames,
    onRename,
    onDelete,
    onEnableBlocked,
}: ProjectActionsMenuProps) => {
    const metadata = { projectId, projectName, projectNames };
    const { menuActions, handleAction } = useProjectMenuActions(
        projectId,
        {
            onRename: () => onRename(metadata),
            onDelete: () => onDelete(metadata),
            onEnableBlocked: () => onEnableBlocked(metadata),
        },
        isPipelineRunning
    );

    return (
        <MenuTrigger>
            <ActionButton
                isQuiet
                UNSAFE_style={{
                    fill: 'var(--spectrum-gray-900)',
                    ...actionButtonStyle,
                }}
                aria-label={'open project options'}
                data-testid={projectId}
            >
                <MoreMenu />
            </ActionButton>
            <Menu onAction={handleAction} UNSAFE_className={classes.actionMenu}>
                {menuActions.map(({ key, label }) => (
                    <Item key={key}>{label}</Item>
                ))}
            </Menu>
        </MenuTrigger>
    );
};

export const MenuActions = ({
    projectId,
    projectName,
    isPipelineRunning,
    actionButtonStyle,
    projectNames,
}: MenuActionsProps) => {
    const [isEnableBlockedDialogOpen, setIsEnableBlockedDialogOpen] = useState(false);
    const deleteProjectDialogState = useOverlayTriggerState({});
    const editProjectNameDialogState = useOverlayTriggerState({});

    return (
        <>
            <ProjectActionsMenu
                projectId={projectId}
                projectName={projectName}
                isPipelineRunning={isPipelineRunning}
                actionButtonStyle={actionButtonStyle}
                projectNames={projectNames}
                onRename={editProjectNameDialogState.open}
                onDelete={deleteProjectDialogState.open}
                onEnableBlocked={() => setIsEnableBlockedDialogOpen(true)}
            />

            <EnablePipelineBlockedDialog
                isOpen={isEnableBlockedDialogOpen}
                onClose={() => setIsEnableBlockedDialogOpen(false)}
            />

            <EditProjectNameDialog
                projectId={projectId}
                projectName={projectName}
                projectNames={projectNames}
                isOpen={editProjectNameDialogState.isOpen}
                onClose={editProjectNameDialogState.close}
            />

            <DeleteProjectDialog
                projectId={projectId}
                projectName={projectName}
                isOpen={deleteProjectDialogState.isOpen}
                onClose={deleteProjectDialogState.close}
            />
        </>
    );
};
