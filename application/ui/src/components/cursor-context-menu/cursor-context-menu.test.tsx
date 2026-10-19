// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode, useRef } from 'react';

import { render, screen } from '@testing-library/react';
import { useOverlayTriggerState } from 'react-stately';

import { CursorContextMenu } from './cursor-context-menu.component';

describe('CursorContextMenu', () => {
    const renderApp = ({
        onOpen = rs.fn(),
        isOpen = false,
        children = <div>Menu Content</div>,
    }: {
        onOpen?: () => void;
        isOpen?: boolean;
        children?: ReactNode;
    }) => {
        const App = () => {
            const triggerRef = useRef<HTMLButtonElement>(null);
            const state = useOverlayTriggerState({ isOpen });

            return (
                <div data-testid='modal' style={{ position: 'relative' }}>
                    <button ref={triggerRef}>trigger</button>
                    <CursorContextMenu state={state} triggerRef={triggerRef} onOpen={onOpen}>
                        {children}
                    </CursorContextMenu>
                </div>
            );
        };
        render(<App />);
    };

    it('does not render the menu when isOpen is false', () => {
        renderApp({ isOpen: false });
        expect(screen.queryByText('Menu Content')).not.toBeInTheDocument();
    });
});
