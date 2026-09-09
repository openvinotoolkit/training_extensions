// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, FileTrigger } from '@geti-ui/ui';

import { acceptedExtensions } from '../../utils';

type AddMediaButtonProps = {
    onFileUpload: (files: File[]) => Promise<void>;
    isDisabled?: boolean;
    testId?: string;
};

export const AddMediaButton = ({
    onFileUpload,
    isDisabled = false,
    testId = 'upload-media-input',
}: AddMediaButtonProps) => {
    const handleFileSelect = async (files: FileList | null) => {
        if (files && files.length > 0) {
            await onFileUpload(Array.from(files));
        }
    };

    return (
        <FileTrigger
            data-testid={testId}
            acceptedFileTypes={[acceptedExtensions]}
            allowsMultiple
            onSelect={handleFileSelect}
        >
            <Button variant={'secondary'} isDisabled={isDisabled} margin={0}>
                Upload media
            </Button>
        </FileTrigger>
    );
};
