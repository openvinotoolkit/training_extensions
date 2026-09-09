// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { BulkLabelsAssignmentDialog } from '../bulk-labels-assignment/bulk-labels-assignment-dialog.component';
import { useUploadFiles } from '../use-upload-files';
import { AddMediaButton } from './add-media-button/add-media-button.component';

export const MediaUpload = ({ testId }: { testId?: string }) => {
    const { isClassification, uploadFiles, uploadMediaLoading, clearFilesForLabelAssignment, filesForLabelAssignment } =
        useUploadFiles();

    return (
        <>
            <AddMediaButton onFileUpload={uploadFiles} isDisabled={uploadMediaLoading} testId={testId} />
            {isClassification && (
                <BulkLabelsAssignmentDialog onClose={clearFilesForLabelAssignment} files={filesForLabelAssignment} />
            )}
        </>
    );
};
