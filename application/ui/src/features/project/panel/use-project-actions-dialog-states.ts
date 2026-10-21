// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { useOverlayTriggerState } from 'react-stately';

import { ProjectActionMetadata } from '../list/menu-actions/menu-actions.component';

export const useProjectActionsDialogStates = () => {
    const projectSelectorState = useOverlayTriggerState({});
    const deleteProjectDialogState = useOverlayTriggerState({});
    const editProjectNameDialogState = useOverlayTriggerState({});
    const enablePipelineBlockedDialogState = useOverlayTriggerState({});
    const [projectActionMetadata, setProjectActionMetadata] = useState<ProjectActionMetadata | null>(null);

    const editProject = (metadata: ProjectActionMetadata) => {
        setProjectActionMetadata(metadata);
        projectSelectorState.close();
        editProjectNameDialogState.open();
    };

    const deleteProject = (metadata: ProjectActionMetadata) => {
        setProjectActionMetadata(metadata);
        projectSelectorState.close();
        deleteProjectDialogState.open();
    };

    const enablePipelineBlocked = (metadata: ProjectActionMetadata) => {
        setProjectActionMetadata(metadata);
        projectSelectorState.close();
        enablePipelineBlockedDialogState.open();
    };

    const closeEnablePipelineBlocked = () => {
        enablePipelineBlockedDialogState.close();
        setProjectActionMetadata(null);
    };

    const closeEditProject = () => {
        editProjectNameDialogState.close();
        setProjectActionMetadata(null);
    };

    const clearProjectActionMetadata = () => {
        setProjectActionMetadata(null);
    };

    return {
        projectActionMetadata,

        editProject,
        isEditProjectNameDialogOpen: editProjectNameDialogState.isOpen,
        closeEditProject,

        isDeleteProjectDialogOpen: deleteProjectDialogState.isOpen,
        deleteProject,
        closeDeleteProject: deleteProjectDialogState.close,

        isEnableBlockedDialogOpen: enablePipelineBlockedDialogState.isOpen,
        enablePipelineBlocked,
        closeEnablePipelineBlocked,

        clearProjectActionMetadata,

        isProjectListOpen: projectSelectorState.isOpen,
        changeProjectListDialogState: projectSelectorState.setOpen,
    };
};
