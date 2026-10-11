// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Context, createContext, ReactNode, useContext, useMemo, useRef } from 'react';

import { $api } from '@/api';
import type { AnnotationDTO, DatasetItem, DatasetSubset, Label, Media } from '@/api/types';
import { InfiniteData, matchQuery, useQueryClient } from '@tanstack/react-query';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { isEqual } from 'lodash-es';

import { getQueryKey } from '../../query-client/query-client';
import { EMPTY_LABEL_ID, isNonEmptyLabel, useProjectLabelsWithEmptyLabel } from '../../shared/labels';
import { isVideoFrame } from '../../shared/media-item-utils';
import type { Annotation, AnnotationLabelRef, Shape } from '../../shared/types';
import { UndoRedoProvider } from '../../shared/undo-redo/undo-redo-provider.component';
import useUndoRedoState from '../../shared/undo-redo/use-undo-redo-state';
import { isNonEmptyArray } from '../../shared/util';
import { mapLocalAnnotationsToServer, mapServerAnnotationsToLocal } from './annotation-mappers';
import type { AnnotatorMode } from './annotator-mode';
import { incrementCachedAnnotatedFrameCount } from './utils';

type DatasetItemsPage = { items: DatasetItem[] };

type AnnotationsContextValue = {
    annotations: Annotation[];
    initialAnnotations: Annotation[];
    initialPredictions: Annotation[];
};

type AnnotationCommands = {
    addAnnotations: (shapes: Shape[], labels: AnnotationLabelRef[]) => string[];
    addAnnotationWithEmptyLabel: (label: Label) => void;
    deleteAnnotations: (annotationIds: string[]) => void;
    updateAnnotations: (updatedAnnotations: Annotation[], labels?: AnnotationLabelRef[]) => void;
    resetAnnotations: () => void;
    replaceAnnotations: (annotations: Annotation[]) => void;
};

type AnnotationSubmission = {
    canSubmit: boolean;
    hasInvalidAnnotation: boolean;
    isSaving: boolean;
    submitAnnotations: (subset: DatasetSubset) => Promise<void>;
    submitPredictions: (subset: DatasetSubset) => Promise<void>;
};

const AnnotationsContext = createContext<AnnotationsContextValue | null>(null);
const AnnotationCommandsContext = createContext<AnnotationCommands | null>(null);
const AnnotationSubmissionContext = createContext<AnnotationSubmission | null>(null);
const IsReadOnlyContext = createContext<boolean | null>(null);

export type AnnotationActionsProviderProps = {
    children: ReactNode;
    initialAnnotationsDTO: AnnotationDTO[];
    initialPredictionsDTO: AnnotationDTO[];
    mediaItem: Media;
    mode: AnnotatorMode;
    isReadOnly?: boolean;
};

const filterOutAnnotationWithEmptyLabel = (annotations: Annotation[]): Annotation[] => {
    return annotations.filter((annotation) => annotation.labels.some((label) => label.id !== EMPTY_LABEL_ID));
};

export const AnnotationActionsProvider = ({
    children,
    initialAnnotationsDTO,
    initialPredictionsDTO,
    mediaItem,
    mode,
    isReadOnly = false,
}: AnnotationActionsProviderProps) => {
    const projectId = useProjectIdentifier();
    const queryClient = useQueryClient();
    const saveMutation = $api.useMutation('post', '/api/projects/{project_id}/dataset/media/{media_id}/annotations', {
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
        },
    });

    const projectLabels = useProjectLabelsWithEmptyLabel();

    const predictions = useMemo(() => {
        return mapServerAnnotationsToLocal(initialPredictionsDTO);
    }, [initialPredictionsDTO]);

    const initialAnnotations = useMemo(() => {
        return mapServerAnnotationsToLocal(initialAnnotationsDTO);
    }, [initialAnnotationsDTO]);

    const [annotations, setAnnotations, undoRedoActions] = useUndoRedoState<Annotation[]>(initialAnnotations);
    const resetHistory = undoRedoActions.reset;

    const prevInitialAnnotationsDTORef = useRef(initialAnnotationsDTO);

    // Reset annotations when source annotations change.
    if (!isEqual(prevInitialAnnotationsDTORef.current, initialAnnotationsDTO)) {
        undoRedoActions.reset(initialAnnotations);
        prevInitialAnnotationsDTORef.current = initialAnnotationsDTO;
    }

    // Updater-form setState keeps the commands stable while annotations change.
    const commands = useMemo<AnnotationCommands>(() => {
        const addAnnotations = (shapes: Shape[], labels: AnnotationLabelRef[]): string[] => {
            const newAnnotations = shapes.map((shape) => ({
                shape,
                id: crypto.randomUUID(),
                labels,
            }));

            setAnnotations((prevAnnotations) => [...prevAnnotations, ...newAnnotations]);

            return newAnnotations.map((annotation) => annotation.id);
        };

        return {
            addAnnotations,
            updateAnnotations: (updatedAnnotations, labels) => {
                if (labels !== undefined) {
                    const idsToUpdate = new Set(updatedAnnotations.map((a) => a.id));

                    setAnnotations((prevAnnotations) =>
                        prevAnnotations.map((annotation) =>
                            idsToUpdate.has(annotation.id) ? { ...annotation, labels } : annotation
                        )
                    );
                } else {
                    const updatedMap = new Map(updatedAnnotations.map((annotation) => [annotation.id, annotation]));

                    setAnnotations((prevAnnotations) =>
                        prevAnnotations.map((annotation) => updatedMap.get(annotation.id) ?? annotation)
                    );
                }
            },
            deleteAnnotations: (annotationIds) => {
                setAnnotations((prevAnnotations) =>
                    prevAnnotations.filter((annotation) => !annotationIds.includes(annotation.id))
                );
            },
            addAnnotationWithEmptyLabel: (emptyLabel) => {
                setAnnotations([]);
                addAnnotations([{ type: 'full_image' }], [{ id: emptyLabel.id }]);
            },
            replaceAnnotations: (newAnnotations) => {
                setAnnotations(() => newAnnotations);
            },
            resetAnnotations: () => {
                resetHistory(initialAnnotations);
            },
        };
    }, [setAnnotations, resetHistory, initialAnnotations]);

    const saveAnnotations = async (annotationsDTO: AnnotationDTO[], subset?: DatasetSubset) => {
        const query = isVideoFrame(mediaItem) ? { frame_index: mediaItem.frame_number } : undefined;

        await saveMutation
            .mutateAsync({
                params: { path: { media_id: mediaItem.id, project_id: projectId }, query },
                body: { annotations: annotationsDTO, subset: subset ?? undefined },
            })
            .then(() => {
                if (isVideoFrame(mediaItem)) {
                    incrementCachedAnnotatedFrameCount(queryClient, mediaItem);
                }
            });

        undoRedoActions.reset(mapServerAnnotationsToLocal(annotationsDTO));
    };

    const submitPredictions = async (subset: DatasetSubset) => {
        const validLabelIds = new Set(projectLabels.map((label) => label.id));
        const serverFormattedAnnotationsWithoutConfidences = mapLocalAnnotationsToServer(predictions, validLabelIds)
            .map(({ confidences, ...restOfAnnotation }) => restOfAnnotation)
            .filter((annotation) => isNonEmptyArray(annotation.labels) && annotation.labels.every(isNonEmptyLabel));

        await saveAnnotations(serverFormattedAnnotationsWithoutConfidences, subset);
    };

    const annotationsToRender = mode === 'annotation' ? annotations : predictions;

    const submitAnnotations = async (subset: DatasetSubset) => {
        const validLabelIds = new Set(projectLabels.map((label) => label.id));
        const filteredAnnotations = filterOutAnnotationWithEmptyLabel(annotations);
        const serverAnnotations = mapLocalAnnotationsToServer(filteredAnnotations, validLabelIds);

        await saveAnnotations(serverAnnotations, subset);
    };

    const hasChangedAnnotations = useMemo(() => {
        const filteredAnnotations = filterOutAnnotationWithEmptyLabel(annotations);
        const currentServerAnnotations = mapLocalAnnotationsToServer(filteredAnnotations);

        return !isEqual(currentServerAnnotations, initialAnnotationsDTO);
    }, [annotations, initialAnnotationsDTO]);

    const hasEmptyLabelSelection = useMemo(() => {
        return annotations.some((annotation) => annotation.labels.some((label) => label.id === EMPTY_LABEL_ID));
    }, [annotations]);

    const hasInvalidAnnotation = useMemo(() => {
        return annotations.some((annotation) => annotation.labels.length === 0);
    }, [annotations]);

    const canSubmit =
        mode === 'prediction'
            ? predictions.length > 0
            : !hasInvalidAnnotation && (hasChangedAnnotations || hasEmptyLabelSelection);
    const isReadOnlyMode = isReadOnly || mode === 'prediction';

    const annotationsValue = useMemo<AnnotationsContextValue>(
        () => ({ annotations: annotationsToRender, initialAnnotations, initialPredictions: predictions }),
        [annotationsToRender, initialAnnotations, predictions]
    );

    const submission: AnnotationSubmission = {
        canSubmit,
        hasInvalidAnnotation,
        isSaving: saveMutation.isPending,
        submitAnnotations,
        submitPredictions,
    };

    return (
        <AnnotationCommandsContext value={commands}>
            <AnnotationsContext value={annotationsValue}>
                <IsReadOnlyContext value={isReadOnlyMode}>
                    <AnnotationSubmissionContext value={submission}>
                        <UndoRedoProvider baseHistory={undoRedoActions}>{children}</UndoRedoProvider>
                    </AnnotationSubmissionContext>
                </IsReadOnlyContext>
            </AnnotationsContext>
        </AnnotationCommandsContext>
    );
};

const useRequiredContext = <T,>(
    context: Context<T | null>,
    hookName: 'useAnnotations' | 'useAnnotationCommands' | 'useIsAnnotatorReadOnly' | 'useAnnotationSubmission'
): T => {
    const value = useContext(context);

    if (value === null) {
        throw new Error(`${hookName} must be used within "AnnotationActionsProvider"`);
    }

    return value;
};

// Re-renders on every annotation change; components that only edit should use useAnnotationCommands.
export const useAnnotations = () => useRequiredContext(AnnotationsContext, 'useAnnotations');

export const useAnnotationCommands = () => useRequiredContext(AnnotationCommandsContext, 'useAnnotationCommands');

export const useIsAnnotatorReadOnly = () => useRequiredContext(IsReadOnlyContext, 'useIsAnnotatorReadOnly');

export const useAnnotationSubmission = () => useRequiredContext(AnnotationSubmissionContext, 'useAnnotationSubmission');
