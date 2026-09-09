// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { render, screen } from '@testing-library/react';

import { IconWrapper } from './icon-wrapper.component';

describe('IconWrapper', () => {
    it('renders children and props correctly', () => {
        const { container } = render(
            <IconWrapper isSelected={true}>
                <span data-testid='test-icon'>Icon</span>
            </IconWrapper>
        );

        expect(screen.getByTestId('test-icon')).toBeInTheDocument();
        expect(screen.getByText('Icon')).toBeInTheDocument();
        expect(container.firstChild).toBeInTheDocument();
    });
});
