// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { render } from 'test-utils/render';

import { openExternalUrl } from '../../platform/open-external-url';
import { HelpMenu } from './help-menu.component';

rs.mock('../../platform/open-external-url', () => ({
    openExternalUrl: rs.fn(),
}));

describe('HelpMenu', () => {
    beforeEach(() => {
        rs.clearAllMocks();
    });

    it.each([
        ['User documentation', 'https://docs.geti.intel.com/'],
        ["What's new", 'https://github.com/open-edge-platform/geti/releases'],
        ['Report an issue', 'https://github.com/open-edge-platform/geti/issues'],
    ])('opens "%s" externally', async (option, url) => {
        render(<HelpMenu />);

        await userEvent.click(screen.getByRole('button', { name: 'Help' }));
        await userEvent.click(await screen.findByRole('menuitem', { name: option }));

        expect(openExternalUrl).toHaveBeenCalledWith(url);
    });
});
