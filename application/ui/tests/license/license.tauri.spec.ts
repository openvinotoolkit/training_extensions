// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { expect, http, test } from '../fixtures';
import { mockTauriRuntime } from '../utils/mock-tauri-runtime';

test.describe('License agreement (Tauri)', () => {
    test.beforeEach(async ({ page }) => {
        // Provide the Tauri IPC bridge that the real desktop webview injects but
        // Playwright's Chromium does not, so the desktop bundle can boot.
        await mockTauriRuntime(page);
    });

    test('shows the Intel and Ultralytics licenses and accepts them', async ({ page, network }) => {
        let licenseAccepted = false;

        network.use(
            http.get('/api/system/info', ({ response }) => {
                return response(200).json({
                    license_accepted: licenseAccepted,
                    platform: 'windows',
                });
            }),
            http.post('/api/license/accept', ({ response }) => {
                licenseAccepted = true;

                return response(200).json({ license_accepted: true });
            }),
            http.get('/api/projects', ({ response }) => {
                return response(200).json([]);
            })
        );

        await test.step('starts on the Intel Simplified Software License', async () => {
            await page.goto('/');

            await expect(page.getByRole('heading', { name: 'License Agreements' })).toBeVisible();
            await expect(page.getByText('1 of 2')).toBeVisible();
            await expect(page.getByRole('heading', { name: 'Intel Simplified Software License' })).toBeVisible();
            await expect(page.getByRole('button', { name: 'Next' })).toBeDisabled();
            await expect(page.getByRole('button', { name: /^2 Ultralytics/ })).toBeDisabled();
        });

        await test.step('agreeing moves on to the Ultralytics notice', async () => {
            await page.getByRole('checkbox', { name: /I have read and agree to the Intel/ }).check();
            await page.getByRole('button', { name: 'Next' }).click();

            await expect(page.getByText('2 of 2')).toBeVisible();
            await expect(page.getByRole('heading', { name: 'Ultralytics AGPL-3.0 License' })).toBeVisible();
            await expect(page.getByRole('button', { name: 'Accept and continue' })).toBeDisabled();
        });

        await test.step('agreeing to the Ultralytics license enables acceptance', async () => {
            await page.getByRole('checkbox', { name: /I have read and agree to the Ultralytics/ }).check();
            await expect(page.getByRole('button', { name: 'Accept and continue' })).toBeEnabled();
        });

        await test.step('accepting redirects to the projects page', async () => {
            await page.getByRole('button', { name: 'Accept and continue' }).click();

            await expect(page.getByRole('heading', { name: 'License Agreements' })).toBeHidden();
            await expect(page).toHaveURL(/\/projects$/);
        });
    });

    test('skips the license screen when it is already accepted', async ({ page, network }) => {
        network.use(
            http.get('/api/system/info', ({ response }) => {
                return response(200).json({
                    license_accepted: true,
                    platform: 'windows',
                });
            }),
            http.get('/api/projects', ({ response }) => {
                return response(200).json([]);
            })
        );

        await page.goto('/');

        await expect(page.getByRole('heading', { name: 'License Agreements' })).toBeHidden();
    });

    test('shows error state when system info is unavailable', async ({ page, network }) => {
        network.use(
            http.get('/api/system/info', ({ response }) => {
                // @ts-expect-error Simulate server error
                return response(500).json({});
            })
        );

        await page.goto('/');

        await expect(page.getByRole('heading', { name: 'Server Error' })).toBeVisible();
        await expect(page.getByRole('button', { name: 'Refresh' })).toBeVisible();
    });
});
