// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, ReactNode, useCallback, useContext, useEffect, useMemo, useState } from 'react';

import { UndoRedoControls } from './undo-redo-actions.interface';

type RegisterToolHistory = (history: UndoRedoControls) => () => void;

const UndoRedoContext = createContext<UndoRedoControls | undefined>(undefined);
const RegisterToolHistoryContext = createContext<RegisterToolHistory | undefined>(undefined);

interface UndoRedoProviderProps {
    children: ReactNode;
    baseHistory: UndoRedoControls;
}

/**
 * Single undo/redo entry point for an annotator session. Undo and redo go to the registered tool
 * history (e.g. an in-progress polygon) while it has entries, and fall back to the base history.
 */
export const UndoRedoProvider = ({ baseHistory, children }: UndoRedoProviderProps) => {
    const [toolHistory, setToolHistory] = useState<UndoRedoControls | null>(null);

    const registerToolHistory = useCallback<RegisterToolHistory>((history) => {
        setToolHistory(history);

        return () => setToolHistory((current) => (current === history ? null : current));
    }, []);

    const canToolUndo = toolHistory?.canUndo ?? false;
    const canToolRedo = toolHistory?.canRedo ?? false;

    const controls = useMemo<UndoRedoControls>(
        () => ({
            canUndo: canToolUndo || baseHistory.canUndo,
            canRedo: canToolRedo || baseHistory.canRedo,
            undo: () => (canToolUndo ? toolHistory?.undo() : baseHistory.undo()),
            redo: () => (canToolRedo ? toolHistory?.redo() : baseHistory.redo()),
        }),
        [canToolUndo, canToolRedo, toolHistory, baseHistory]
    );

    return (
        <RegisterToolHistoryContext value={registerToolHistory}>
            <UndoRedoContext value={controls}>{children}</UndoRedoContext>
        </RegisterToolHistoryContext>
    );
};

export const useUndoRedo = (): UndoRedoControls => {
    const context = useContext(UndoRedoContext);

    if (context === undefined) {
        throw new Error('useUndoRedo must be used within an UndoRedoProvider');
    }

    return context;
};

// `history` must keep a stable identity between changes, or it re-registers on every render.
export const useRegisterToolHistory = (history: UndoRedoControls): void => {
    const register = useContext(RegisterToolHistoryContext);

    if (register === undefined) {
        throw new Error('useRegisterToolHistory must be used within an UndoRedoProvider');
    }

    useEffect(() => register(history), [register, history]);
};
