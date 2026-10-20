// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import React from 'react';

import { screen } from '@testing-library/react';
import { getMockedLabel } from 'mocks/mock-labels';
import * as actualRecharts from 'recharts' with { rstest: 'importActual' };
import { render } from 'test-utils/render';

import { useProjectLabelsWithEmptyLabel } from '../../../../../shared/labels';
import * as actualLabels from '../../../../../shared/labels' with { rstest: 'importActual' };
import { DatasetLabelsChart } from './dataset-labels-chart.component';

rs.mock('recharts', () => {
    return {
        ...actualRecharts,
        ResponsiveContainer: ({ children }: { children: React.ReactElement }) =>
            React.cloneElement(children, { width: 800, height: 200 } as React.HTMLAttributes<unknown>),
    };
});

const mockLabels = [
    getMockedLabel({ id: 'label-1', name: 'Fish-1' }),
    getMockedLabel({ id: 'label-2', name: 'Fish-2' }),
    getMockedLabel({ id: 'label-3', name: 'Fish-3' }),
    getMockedLabel({ id: 'label-4', name: 'Fish-4' }),
    getMockedLabel({ id: 'label-5', name: 'Fish-5' }),
    getMockedLabel({ id: 'label-6', name: 'Fish-6' }),
    getMockedLabel({ id: 'label-7', name: 'Fish-7' }),
    getMockedLabel({ id: 'label-8', name: 'Fish-8' }),
    getMockedLabel({ id: 'label-9', name: 'Fish-9' }),
    getMockedLabel({ id: 'label-10', name: 'Fish-10' }),
];

rs.mock('../../../../../shared/labels', () => {
    return {
        ...actualLabels,
        useProjectLabelsWithEmptyLabel: rs.fn(),
    };
});

describe('DatasetLabelsChart', () => {
    beforeEach(() => {
        rs.mocked(useProjectLabelsWithEmptyLabel).mockReturnValue(mockLabels);
    });

    it('renders all label names with multiple items', () => {
        const instancesPerLabel = mockLabels.map(({ id }) => ({ label_id: id, instances: 10 }));
        render(<DatasetLabelsChart totalItems={instancesPerLabel.length * 2} instancesPerLabel={instancesPerLabel} />);

        mockLabels.forEach(({ name }) => {
            expect(screen.getByText(name)).toBeVisible();
        });
    });

    it('maps null label_id to the empty label', async () => {
        const emptyLabel = getMockedLabel({ id: 'empty-label', name: 'No object' });
        const labelsWithEmpty = [...mockLabels, emptyLabel];

        rs.mocked(useProjectLabelsWithEmptyLabel).mockReturnValue(labelsWithEmpty);

        const instancesPerLabel = [{ label_id: null, instances: 3 }];
        render(<DatasetLabelsChart totalItems={10} instancesPerLabel={instancesPerLabel} />);

        expect(screen.getByText('No object')).toBeVisible();
    });
});
