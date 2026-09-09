// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import type { ModelArchitecture } from '@/api/types';

import {
    useGetTimmFamilies,
    useGetTimmManifest,
    useGetTimmPretrainedTags,
    useGetTimmVariants,
} from '../api/use-get-timm-catalog';

export type TimmModelSelection = {
    timmFamilies: string[];
    timmVariants: string[];
    timmPretrainedTags: string[];

    selectedTimmFamily: string | null;
    onSelectTimmFamily: (family: string) => void;

    selectedTimmVariant: string | null;
    onSelectTimmVariant: (variant: string) => void;

    selectedTimmPretrainedTag: string | null;
    onSelectTimmPretrainedTag: (pretrainedTag: string) => void;

    timmModelArchitecture: ModelArchitecture | undefined;
    isLoadingTimmModelArchitecture: boolean;
};

// Family -> variant -> pretrained tag
export const useTimmModelSelection = (isEnabled: boolean): TimmModelSelection => {
    const [family, setFamily] = useState<string | null>(null);
    const [variant, setVariant] = useState<string | null>(null);
    const [pretrainedTag, setPretrainedTag] = useState<string | null>(null);

    const selectedTimmFamily = isEnabled ? family : null;

    const { data: timmFamilies = [] } = useGetTimmFamilies(isEnabled);
    const { data: timmVariants = [] } = useGetTimmVariants(selectedTimmFamily);

    const selectedTimmVariant = variant ?? timmVariants?.at(0) ?? null;

    const { data: timmPretrainedTags = [] } = useGetTimmPretrainedTags(selectedTimmFamily, selectedTimmVariant);

    const selectedTimmPretrainedTag = pretrainedTag ?? timmPretrainedTags?.at(0) ?? null;

    const { data: timmModelArchitecture, isFetching: isLoadingTimmModelArchitecture } = useGetTimmManifest(
        selectedTimmFamily,
        selectedTimmVariant,
        selectedTimmPretrainedTag
    );

    const onSelectTimmFamily = (nextFamily: string) => {
        setFamily(nextFamily);
        setVariant(null);
        setPretrainedTag(null);
    };

    const onSelectTimmVariant = (nextVariant: string) => {
        setVariant(nextVariant);
        setPretrainedTag(null);
    };

    return {
        timmFamilies,
        timmVariants,
        timmPretrainedTags,

        selectedTimmFamily,
        onSelectTimmFamily,

        selectedTimmVariant,
        onSelectTimmVariant,

        selectedTimmPretrainedTag,
        onSelectTimmPretrainedTag: setPretrainedTag,

        timmModelArchitecture,
        isLoadingTimmModelArchitecture,
    };
};
