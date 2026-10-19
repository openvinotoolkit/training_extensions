// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, Dispatch, ReactNode, useContext, useState } from 'react';

import { OverlayTriggerState, useOverlayTriggerState } from 'react-stately';

import { ImportDatasetAsNewProjectState } from '../../../shared/import-dataset-state';

type ImportDatasetDialogStateContextProps = {
    datasetImportDialogState: OverlayTriggerState;
    currentStagedId: string | null;
    setCurrentStagedId: Dispatch<React.SetStateAction<string | null>>;
    currentStep: ImportDatasetAsNewProjectState;
    setCurrentStep: Dispatch<React.SetStateAction<ImportDatasetAsNewProjectState>>;
};

const ImportDatasetDialogContext = createContext<ImportDatasetDialogStateContextProps | undefined>(undefined);

export const ImportDatasetDialogProvider = ({ children }: { children: ReactNode }) => {
    const datasetImportDialogState = useOverlayTriggerState({});
    const [currentStep, setCurrentStep] = useState<ImportDatasetAsNewProjectState>('uploading');
    const [currentStagedId, setCurrentStagedId] = useState<string | null>(null);

    return (
        <ImportDatasetDialogContext.Provider
            value={{ datasetImportDialogState, currentStagedId, setCurrentStagedId, currentStep, setCurrentStep }}
        >
            {children}
        </ImportDatasetDialogContext.Provider>
    );
};

export const useImportDatasetDialog = () => {
    const context = useContext(ImportDatasetDialogContext);

    if (context === undefined) {
        throw new Error('useImportDatasetDialog was used outside of ImportDatasetDialogProvider');
    }

    return context;
};
