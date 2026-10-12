// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useMemo } from 'react';

import { $api } from '@/api';
import type { AnnotationDTO, DatasetItem, DatasetSubset, Media } from '@/api/types';
import { InfiniteData, matchQuery, useQueryClient } from '@tanstack/react-query';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { isEqual } from 'lodash-es';

import {
    useAnnotationCommands,
    useAnnotations,
} from '../../../../modules/annotator/annotation-document-provider.component';
import {
    mapLocalAnnotationsToServer,
    mapServerAnnotationsToLocal,
} from '../../../../modules/annotator/annotation-mappers';
import { getQueryKey } from '../../../../query-client/query-client';
import { isEmptyLabel, isNonEmptyLabel, useProjectLabelsWithEmptyLabel } from '../../../../shared/labels';
import { isVideoFrame } from '../../../../shared/media-item-utils';
import type { Annotation } from '../../../../shared/types';
import { isNonEmptyArray } from '../../../../shared/util';
import { incrementCachedAnnotatedFrameCount } from './increment-cached-annotated-frame-count';

type DatasetItemsPage = { items: DatasetItem[] };

const filterOutAnnotationWithEmptyLabel = (annotations: Annotation[]): Annotation[] => {
    return annotations.filter((annotation) => annotation.labels.some(isNonEmptyLabel));
};

const useSaveAnnotationsMutation = (mediaItem: Media) => {
    const projectId = useProjectIdentifier();
    const queryClient = useQueryClient();

    return $api.useMutation('post', '/api/projects/{project_id}/dataset/media/{media_id}/annotations', {
        meta: {
            invalidateQueries: [
                [
                    'get',
                    '/api/projects/{project_id}/dataset/media/{media_id}/annotations',
                    { params: { path: { project_id: projectId, media_id: mediaItem.id } } },
                ],
                [
                    'get',
                    '/api/projects/{project_id}/dataset/items/{dataset_item_id}',
                    { params: { path: { project_id: projectId, dataset_item_id: mediaItem.id } } },
                ],
                [
                    'get',
                    '/api/projects/{project_id}/dataset/media/{media_id}/frames',
                    { params: { path: { project_id: projectId, media_id: mediaItem.id } } },
                ],
            ],
        },
        onSuccess: (data) => {
            // The dataset item list backs the sidebar's review-status badges. Instead of
            // invalidating it over the network (which would refetch every page the user has
            // already scrolled through - potentially dozens of requests on every single
            // submit), patch the affected item directly in the already-cached pages.
            const datasetItemsQueryKey = getQueryKey([
                'get',
                '/api/projects/{project_id}/dataset/items',
                { params: { path: { project_id: projectId } } },
            ]);

            queryClient.setQueriesData<InfiniteData<DatasetItemsPage>>(
                { predicate: (query) => matchQuery({ queryKey: datasetItemsQueryKey }, query) },
                (previousData) => {
                    if (previousData === undefined) {
                        return previousData;
                    }

                    return {
                        ...previousData,
                        pages: previousData.pages.map((page) => ({
                            ...page,
                            items: page.items.map((item) =>
                                item.id === data.media_id
                                    ? { ...item, subset: data.subset, user_reviewed: data.user_reviewed }
                                    : item
                            ),
                        })),
                    };
                }
            );

            if (isVideoFrame(mediaItem)) {
                incrementCachedAnnotatedFrameCount(queryClient, mediaItem);
            }
        },
    });
};

export const useSubmitAnnotations = ({
    mediaItem,
    hasSubsetChanged = false,
}: {
    mediaItem: Media;
    hasSubsetChanged?: boolean;
}) => {
    const projectId = useProjectIdentifier();
    const projectLabels = useProjectLabelsWithEmptyLabel();
    const { mode, annotations, initialAnnotations, initialPredictions } = useAnnotations();
    const { resetAnnotations } = useAnnotationCommands();
    const saveMutation = useSaveAnnotationsMutation(mediaItem);

    // In prediction mode `annotations` holds the predictions, so the user's edits are only read in annotation mode.
    const isPredictionMode = mode === 'prediction';

    const saveAnnotations = async (annotationsDTO: AnnotationDTO[], subset: DatasetSubset) => {
        const query = isVideoFrame(mediaItem) ? { frame_index: mediaItem.frame_number } : undefined;

        await saveMutation.mutateAsync({
            params: { path: { media_id: mediaItem.id, project_id: projectId }, query },
            body: { annotations: annotationsDTO, subset },
        });

        resetAnnotations(mapServerAnnotationsToLocal(annotationsDTO));
    };

    const submit = async (subset: DatasetSubset) => {
        const validLabelIds = new Set(projectLabels.map((label) => label.id));

        if (isPredictionMode) {
            const predictionsWithoutConfidences = mapLocalAnnotationsToServer(initialPredictions, validLabelIds)
                .map(({ confidences, ...restOfAnnotation }) => restOfAnnotation)
                .filter((annotation) => isNonEmptyArray(annotation.labels) && annotation.labels.every(isNonEmptyLabel));

            await saveAnnotations(predictionsWithoutConfidences, subset);

            return;
        }

        await saveAnnotations(
            mapLocalAnnotationsToServer(filterOutAnnotationWithEmptyLabel(annotations), validLabelIds),
            subset
        );
    };

    const initialServerAnnotations = useMemo(
        () => mapLocalAnnotationsToServer(initialAnnotations),
        [initialAnnotations]
    );

    // Only the current side drops empty-label annotations: removing a saved "empty" label counts as a change.
    const hasChangedAnnotations = useMemo(
        () =>
            !isEqual(
                mapLocalAnnotationsToServer(filterOutAnnotationWithEmptyLabel(annotations)),
                initialServerAnnotations
            ),
        [annotations, initialServerAnnotations]
    );

    const hasEmptyLabelSelection = annotations.some((annotation) => annotation.labels.some(isEmptyLabel));
    const hasInvalidAnnotation = !isPredictionMode && annotations.some((annotation) => annotation.labels.length === 0);

    const canSubmit = isPredictionMode
        ? initialPredictions.length > 0
        : !hasInvalidAnnotation && (hasChangedAnnotations || hasEmptyLabelSelection || hasSubsetChanged);

    return {
        canSubmit,
        hasInvalidAnnotation,
        isSaving: saveMutation.isPending,
        submit,
    };
};
