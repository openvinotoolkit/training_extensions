// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useSearchParams } from 'react-router-dom';

import { isNonEmptyString } from '../shared/util';

export const DATASET_VIEW_ID_PARAM = 'datasetViewId';
export const ENTIRE_DATASET_VIEW_ID = null;

export const useDatasetViewId = () => {
    const [searchParams, setSearchParams] = useSearchParams();

    const rawValue = searchParams.get(DATASET_VIEW_ID_PARAM);
    const datasetViewId = isNonEmptyString(rawValue) ? rawValue : ENTIRE_DATASET_VIEW_ID;

    const setDatasetViewId = (id: string | null) => {
        setSearchParams(
            (prev) => {
                if (id === ENTIRE_DATASET_VIEW_ID) {
                    prev.delete(DATASET_VIEW_ID_PARAM);
                } else {
                    prev.set(DATASET_VIEW_ID_PARAM, id);
                }

                return prev;
            },
            { replace: true }
        );
    };

    return [datasetViewId, setDatasetViewId] as const;
};
