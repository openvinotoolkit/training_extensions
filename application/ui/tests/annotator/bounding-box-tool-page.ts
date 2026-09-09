// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Page } from '@playwright/test';

import { Rect } from '../../src/shared/types';
import { clickAndMove, withRelative } from '../utils/mouse';

export class BoundingBoxToolPage {
    constructor(private page: Page) {}

    async drawBoundingBox({ x, y, width, height }: Omit<Rect, 'type'>) {
        const relative = await withRelative(this.page);
        const startPoint = relative(x, y);
        const endPoint = relative(x + width, y + height);

        await clickAndMove(this.page, startPoint, endPoint);
    }

    getTool() {
        return this.page.getByRole('button', { name: 'Bounding box' });
    }

    async selectTool() {
        await this.getTool().click();
    }
}
