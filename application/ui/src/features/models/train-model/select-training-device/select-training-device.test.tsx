// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { TrainingDevice } from '@/api/types';
import { fireEvent, screen } from '@testing-library/react';
import { render } from 'test-utils/render';

import * as actualTrainModelProvider from '../train-model-provider.component' with { rstest: 'importActual' };
import { SelectTrainingDevice } from './select-training-device.component';

const mockOnSelectTrainingDevice = rs.hoisted(() => rs.fn());

const getMockedTrainingDevice = (device: Partial<TrainingDevice> = {}): TrainingDevice => ({
    type: 'cpu',
    name: 'CPU',
    memory: null,
    index: null,
    ...device,
});

const cpuDevice = getMockedTrainingDevice({ type: 'cpu', name: 'CPU', memory: null, index: null });
const gpuDevice = getMockedTrainingDevice({ type: 'cuda', name: 'NVIDIA GPU', memory: 8_000_000_000, index: null });
const gpuDeviceWithIndex = getMockedTrainingDevice({
    type: 'cuda',
    name: 'NVIDIA GPU',
    memory: 8_000_000_000,
    index: 0,
});

const mockTrainingDevices = [cpuDevice, gpuDevice, gpuDeviceWithIndex];

rs.mock('../train-model-provider.component', () => ({
    ...actualTrainModelProvider,
    useTrainModelState: () => ({
        trainingDevices: mockTrainingDevices,
        onSelectTrainingDevice: mockOnSelectTrainingDevice,
        selectedTrainingDevice: 'cpu',
    }),
}));

const openPicker = () => {
    fireEvent.click(screen.getByRole('button'));
};

describe('SelectTrainingDevice', () => {
    beforeEach(() => {
        mockOnSelectTrainingDevice.mockReset();
    });

    it('renders the picker with label "Select training device"', () => {
        render(<SelectTrainingDevice />);

        expect(screen.getByRole('button', { name: /Select training device/i })).toBeInTheDocument();
    });

    it('renders CPU device with just the name (no memory, no index)', async () => {
        render(<SelectTrainingDevice />);

        openPicker();

        expect(await screen.findByRole('option', { name: 'CPU' })).toBeInTheDocument();
    });

    it('renders GPU device with memory formatted as "Name (X GB)"', async () => {
        render(<SelectTrainingDevice />);

        openPicker();

        expect(await screen.findByRole('option', { name: 'NVIDIA GPU (8 GB)' })).toBeInTheDocument();
    });

    it('renders GPU device with memory and index as "Name (X GB) [0]"', async () => {
        render(<SelectTrainingDevice />);

        openPicker();

        expect(await screen.findByRole('option', { name: 'NVIDIA GPU (8 GB) [0]' })).toBeInTheDocument();
    });

    it('uses type as key for CPU (no index)', async () => {
        render(<SelectTrainingDevice />);

        openPicker();

        const option = await screen.findByRole('option', { name: 'CPU' });
        expect(option).toHaveAttribute('data-key', 'cpu');
    });

    it('uses "type-index" as key for GPU with index', async () => {
        render(<SelectTrainingDevice />);

        openPicker();

        const option = await screen.findByRole('option', { name: 'NVIDIA GPU (8 GB) [0]' });
        expect(option).toHaveAttribute('data-key', 'cuda-0');
    });

    it('calls onSelectTrainingDevice with the string key when a valid key is selected', async () => {
        render(<SelectTrainingDevice />);

        openPicker();

        const option = await screen.findByRole('option', { name: 'NVIDIA GPU (8 GB)' });
        fireEvent.click(option);

        expect(mockOnSelectTrainingDevice).toHaveBeenCalledWith('cuda');
    });
});
