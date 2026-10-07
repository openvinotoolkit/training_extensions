// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Context, createContext, ReactNode, useContext, useMemo, useRef } from 'react';

import type { AnnotationDTO, Label } from '@/api/types';
import { isEqual } from 'lodash-es';

import type { Annotation, AnnotationLabelRef, Shape } from '../../shared/types';
import { UndoRedoProvider } from '../../shared/undo-redo/undo-redo-provider.component';
import useUndoRedoState from '../../shared/undo-redo/use-undo-redo-state';
import { mapServerAnnotationsToLocal } from './annotation-mappers';
import type { AnnotatorMode } from './annotator-mode';

type AnnotationsContextValue = {
    mode: AnnotatorMode;
    // What is on screen: the user's edits in annotation mode, the predictions in prediction mode.
    annotations: Annotation[];
    initialAnnotations: Annotation[];
    initialPredictions: Annotation[];
};

type AnnotationCommands = {
    // Adding any annotation replaces an existing full-image annotation.
    addAnnotations: (shapes: Shape[], labels: AnnotationLabelRef[]) => string[];
    addAnnotationWithEmptyLabel: (label: Label) => void;
    deleteAnnotations: (annotationIds: string[]) => void;
    updateAnnotations: (updatedAnnotations: Annotation[], labels?: AnnotationLabelRef[]) => void;
    // Starts a fresh history at `annotations`, or at the initial annotations when omitted.
    resetAnnotations: (annotations?: Annotation[]) => void;
    replaceAnnotations: (annotations: Annotation[]) => void;
};

const AnnotationsContext = createContext<AnnotationsContextValue | null>(null);
const AnnotationCommandsContext = createContext<AnnotationCommands | null>(null);
const IsReadOnlyContext = createContext<boolean | null>(null);

const createAnnotations = (shapes: Shape[], labels: AnnotationLabelRef[]): Annotation[] =>
    shapes.map((shape) => ({ shape, id: crypto.randomUUID(), labels }));

const isFullImageAnnotation = (annotation: Annotation) => annotation.shape.type === 'full_image';

export type AnnotationDocumentProviderProps = {
    children: ReactNode;
    initialAnnotationsDTO: AnnotationDTO[];
    initialPredictionsDTO: AnnotationDTO[];
    mode: AnnotatorMode;
    isReadOnly?: boolean;
};

// Local, undoable annotation state for one media item. Persisting it is up to the consuming feature.
export const AnnotationDocumentProvider = ({
    children,
    initialAnnotationsDTO,
    initialPredictionsDTO,
    mode,
    isReadOnly = false,
}: AnnotationDocumentProviderProps) => {
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
        return {
            addAnnotations: (shapes, labels) => {
                const newAnnotations = createAnnotations(shapes, labels);

                setAnnotations((prevAnnotations) => [
                    ...prevAnnotations.filter((annotation) => !isFullImageAnnotation(annotation)),
                    ...newAnnotations,
                ]);

                return newAnnotations.map((annotation) => annotation.id);
            },
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
                const idsToDelete = new Set(annotationIds);

                setAnnotations((prevAnnotations) =>
                    prevAnnotations.filter((annotation) => !idsToDelete.has(annotation.id))
                );
            },
            addAnnotationWithEmptyLabel: (emptyLabel) => {
                setAnnotations(createAnnotations([{ type: 'full_image' }], [{ id: emptyLabel.id }]));
            },
            replaceAnnotations: (newAnnotations) => {
                setAnnotations(() => newAnnotations);
            },
            resetAnnotations: (newAnnotations = initialAnnotations) => {
                resetHistory(newAnnotations);
            },
        };
    }, [setAnnotations, resetHistory, initialAnnotations]);

    const annotationsToRender = mode === 'annotation' ? annotations : predictions;
    const isReadOnlyMode = isReadOnly || mode === 'prediction';

    const annotationsValue = useMemo<AnnotationsContextValue>(
        () => ({ mode, annotations: annotationsToRender, initialAnnotations, initialPredictions: predictions }),
        [mode, annotationsToRender, initialAnnotations, predictions]
    );

    return (
        <AnnotationCommandsContext value={commands}>
            <AnnotationsContext value={annotationsValue}>
                <IsReadOnlyContext value={isReadOnlyMode}>
                    <UndoRedoProvider baseHistory={undoRedoActions}>{children}</UndoRedoProvider>
                </IsReadOnlyContext>
            </AnnotationsContext>
        </AnnotationCommandsContext>
    );
};

const useRequiredContext = <T,>(context: Context<T | null>, hookName: string): T => {
    const value = useContext(context);

    if (value === null) {
        throw new Error(`${hookName} must be used within "AnnotationDocumentProvider"`);
    }

    return value;
};

// Re-renders on every annotation change; components that only edit should use useAnnotationCommands.
export const useAnnotations = () => useRequiredContext(AnnotationsContext, 'useAnnotations');

export const useAnnotationCommands = () => useRequiredContext(AnnotationCommandsContext, 'useAnnotationCommands');

export const useIsAnnotatorReadOnly = () => useRequiredContext(IsReadOnlyContext, 'useIsAnnotatorReadOnly');
