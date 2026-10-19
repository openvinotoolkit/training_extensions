// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../api/utils';
import { server } from '../../msw-node-setup';
import { InferenceDevices } from './inference-devices.component';

const mockDevices = [
    { type: 'cpu' as const, name: 'CPU', memory: null, index: null },
    { type: 'xpu' as const, name: 'XPU', memory: null, index: null },
];

describe('InferenceDevices', () => {
    beforeEach(() => {
        server.use(http.get('/api/system/devices/inference', () => HttpResponse.json(mockDevices)));
    });

    it('renders picker items from the devices API', async () => {
        const onSelectionChange = rs.fn();
        render(<InferenceDevices selectedKey='cpu' onSelectionChange={onSelectionChange} label='device picker' />);

        const picker = await screen.findByRole('button', { name: /device picker/i });
        expect(picker).toBeInTheDocument();

        await userEvent.click(picker);

        expect(await screen.findByRole('option', { name: /CPU/i })).toBeInTheDocument();
        expect(screen.getByRole('option', { name: /XPU/i })).toBeInTheDocument();
    });

    it('calls onSelectionChange with the device key when a different device is selected', async () => {
        const onSelectionChange = rs.fn();
        render(<InferenceDevices selectedKey='cpu' onSelectionChange={onSelectionChange} label='device picker' />);

        await userEvent.click(await screen.findByRole('button', { name: /device picker/i }));
        await userEvent.click(await screen.findByRole('option', { name: /XPU/i }));

        await waitFor(() => {
            expect(onSelectionChange).toHaveBeenCalledWith('xpu');
        });
    });

    it('does not call onSelectionChange when the already-selected key is clicked again', async () => {
        const onSelectionChange = rs.fn();
        render(<InferenceDevices selectedKey='cpu' onSelectionChange={onSelectionChange} label='device picker' />);

        await userEvent.click(await screen.findByRole('button', { name: /device picker/i }));
        await userEvent.click(await screen.findByRole('option', { name: /CPU/i }));

        expect(onSelectionChange).not.toHaveBeenCalled();
    });

    it('generates composite key type-index for devices with index', async () => {
        const devicesWithIndex = [
            { type: 'cpu' as const, name: 'CPU', memory: null, index: null },
            { type: 'xpu' as const, name: 'XPU', memory: null, index: 0 },
        ];
        server.use(http.get('/api/system/devices/inference', () => HttpResponse.json(devicesWithIndex)));

        const onSelectionChange = rs.fn();
        render(<InferenceDevices selectedKey='cpu' onSelectionChange={onSelectionChange} label='device picker' />);

        await userEvent.click(await screen.findByRole('button', { name: /device picker/i }));
        await userEvent.click(await screen.findByRole('option', { name: /XPU/i }));

        await waitFor(() => {
            expect(onSelectionChange).toHaveBeenCalledWith('xpu-0');
        });
    });
});
