// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { expect, Page } from '@playwright/test';

import type { Rect } from '../../src/shared/types';
import { clickAndMove, withRelative } from '../utils/mouse';

export class SSIMToolPage {
    constructor(private page: Page) {}

    async drawTemplate({ x, y, width, height }: Omit<Rect, 'type'>) {
        const relative = await withRelative(this.page);
        const startPoint = relative(x, y);
        const endPoint = relative(x + width, y + height);

        await clickAndMove(this.page, startPoint, endPoint);
    }

    getTool() {
        return this.page.getByRole('button', { name: 'Detection assistant' });
    }

    async selectTool() {
        await this.getTool().click();
        // The drawing canvas only renders once the SSIM worker has finished loading.
        await expect(this.page.getByLabel('tool', { exact: true })).toBeVisible({ timeout: 30000 });
    }
}
