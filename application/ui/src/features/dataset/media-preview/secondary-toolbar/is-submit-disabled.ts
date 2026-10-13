// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

type IsSubmitDisabledParams = {
    isLoadingPredictions: boolean;
    canSubmit: boolean;
    isSaving: boolean;
};

export const getIsSubmitDisabled = ({ isLoadingPredictions, canSubmit, isSaving }: IsSubmitDisabledParams): boolean =>
    !canSubmit || isSaving || isLoadingPredictions;
