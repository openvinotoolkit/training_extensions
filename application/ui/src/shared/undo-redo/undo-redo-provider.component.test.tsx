// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { fireEvent, renderHook, screen } from '@testing-library/react';
import { render } from 'test-utils/render';

import { UndoRedoProvider, useRegisterToolHistory, useUndoRedo } from './undo-redo-provider.component';
import useUndoRedoState from './use-undo-redo-state';

const ToolHistory = () => {
    const [value, setValue, history] = useUndoRedoState(0);

    useRegisterToolHistory(history);

    return (
        <div>
            <span aria-label='Tool value'>{value}</span>
            <button onClick={() => setValue((prev) => prev + 1)}>Increase tool</button>
        </div>
    );
};

const Toolbar = () => {
    const { undo, redo, canUndo, canRedo } = useUndoRedo();

    return (
        <div>
            <button onClick={undo} disabled={!canUndo}>
                Undo
            </button>
            <button onClick={redo} disabled={!canRedo}>
                Redo
            </button>
        </div>
    );
};

const Session = () => {
    const [value, setValue, history] = useUndoRedoState(0);
    const [isToolMounted, setIsToolMounted] = useState(true);

    return (
        <UndoRedoProvider baseHistory={history}>
            <span aria-label='Base value'>{value}</span>
            <button onClick={() => setValue((prev) => prev + 1)}>Increase base</button>
            <button onClick={() => setIsToolMounted(false)}>Unmount tool</button>
            {/* The toolbar is rendered before the tool on purpose: routing must not depend on tree order. */}
            <Toolbar />
            {isToolMounted && <ToolHistory />}
        </UndoRedoProvider>
    );
};

const click = (name: string) => fireEvent.click(screen.getByRole('button', { name }));
const baseValue = () => screen.getByLabelText('Base value');
const toolValue = () => screen.getByLabelText('Tool value');

describe('UndoRedoProvider', () => {
    it('undoes the tool history first, then falls back to the base history', () => {
        render(<Session />);

        click('Increase base');
        click('Increase tool');
        click('Increase tool');

        click('Undo');
        expect(toolValue()).toHaveTextContent('1');
        expect(baseValue()).toHaveTextContent('1');

        click('Undo');
        expect(toolValue()).toHaveTextContent('0');
        expect(baseValue()).toHaveTextContent('1');

        click('Undo');
        expect(toolValue()).toHaveTextContent('0');
        expect(baseValue()).toHaveTextContent('0');
        expect(screen.getByRole('button', { name: 'Undo' })).toBeDisabled();
    });

    it('redoes the tool history first, then falls back to the base history', () => {
        render(<Session />);

        click('Increase base');
        click('Increase tool');
        click('Undo');
        click('Undo');
        expect(toolValue()).toHaveTextContent('0');
        expect(baseValue()).toHaveTextContent('0');

        click('Redo');
        expect(toolValue()).toHaveTextContent('1');
        expect(baseValue()).toHaveTextContent('0');

        click('Redo');
        expect(baseValue()).toHaveTextContent('1');
        expect(screen.getByRole('button', { name: 'Redo' })).toBeDisabled();
    });

    it('only uses the base history once the tool unmounts', () => {
        render(<Session />);

        click('Increase tool');
        click('Unmount tool');
        expect(screen.getByRole('button', { name: 'Undo' })).toBeDisabled();

        click('Increase base');
        click('Undo');
        expect(baseValue()).toHaveTextContent('0');
    });

    it('throws when a tool history is registered outside the provider', () => {
        vi.spyOn(console, 'error').mockImplementation(() => {});
        const history = { canUndo: false, canRedo: false, undo: vi.fn(), redo: vi.fn() };

        expect(() => renderHook(() => useRegisterToolHistory(history))).toThrow(
            'useRegisterToolHistory must be used within an UndoRedoProvider'
        );
    });
});
