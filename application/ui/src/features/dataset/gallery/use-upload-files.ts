// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { useProject } from 'hooks/api/project.hook';

import { isClassificationTask } from '../../project/task-type-guards';
import { useMediaUpload } from '../api/use-media-upload';
import { useIsUploading } from '../providers/media-upload-provider.component';
import { isVideoFile } from './utils';

export const useUploadFiles = () => {
    const { data: project } = useProject();
    const { uploadMedia } = useMediaUpload();
    const isUploading = useIsUploading();

    const [filesForLabelAssignment, setFilesForLabelAssignment] = useState<File[]>([]);
    const isClassification = isClassificationTask(project.task.task_type);

    const handleFileUpload = async (files: File[]) => {
        if (files.length === 0) {
            return;
        }

        if (isClassification && !files.every(isVideoFile)) {
            setFilesForLabelAssignment(files);
        } else {
            await uploadMedia(files);
        }
    };

    const clearFilesForLabelAssignment = () => {
        setFilesForLabelAssignment([]);
    };

    return {
        uploadMedia,
        uploadMediaLoading: isUploading,
        uploadFiles: handleFileUpload,
        isClassification,
        filesForLabelAssignment,
        clearFilesForLabelAssignment,
    };
};
