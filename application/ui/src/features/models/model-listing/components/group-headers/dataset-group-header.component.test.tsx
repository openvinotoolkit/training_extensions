// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';
import { localISOString } from 'test-utils/local-iso-string';
import { render } from 'test-utils/render';

import type { DatasetGroup } from '../../types';
import { DatasetGroupHeader } from './dataset-group-header.component';

const mockDataset: DatasetGroup = {
    id: 'dataset-123',
    name: 'Test Dataset',
    createdAt: localISOString(2025, 10, 1, 11, 7),
    labelCount: 5,
    imageCount: 100,
    trainingSubsets: {
        training: 70,
        validation: 20,
        testing: 10,
    },
    filesDeleted: false,
};

describe('DatasetGroupHeader', () => {
    it('renders the formatted creation date', () => {
        render(<DatasetGroupHeader dataset={mockDataset} />);

        expect(screen.getByText('Created Oct 01, 2025, 11:07 AM')).toBeInTheDocument();
    });

    it('renders "-" when createdAt is null', () => {
        render(<DatasetGroupHeader dataset={{ ...mockDataset, createdAt: null }} />);

        expect(screen.getByText('-')).toBeInTheDocument();
    });

    it('renders "-" when createdAt is an invalid date string', () => {
        render(<DatasetGroupHeader dataset={{ ...mockDataset, createdAt: 'not a date' }} />);

        expect(screen.getByText('-')).toBeInTheDocument();
    });
});
