// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { TaskType } from '@/api/types';
import { fireEvent, screen } from '@testing-library/react';
import { render } from 'test-utils/render';

import { TaskSelection } from './task-selection.component';

describe('TaskSelection', () => {
    const App = ({ selectedTask = 'detection', setSelectedTask = rs.fn() }) => {
        return <TaskSelection selectedTask={selectedTask as TaskType} setSelectedTask={setSelectedTask} />;
    };

    it('renders all task options', () => {
        render(<App />);

        expect(screen.getByLabelText('Task option: Object Detection')).toBeInTheDocument();
        expect(screen.getByLabelText('Task option: Instance Segmentation')).toBeInTheDocument();
        expect(screen.getByLabelText('Task option: Image Classification')).toBeInTheDocument();

        expect(screen.getByText('Best for: Counting, Tracking')).toBeInTheDocument();
        expect(screen.getByText('Best for: Measurement, Odd shapes')).toBeInTheDocument();
        expect(screen.getByText('Best for: Filtering, Content Moderation')).toBeInTheDocument();
    });

    it('selects the first task by default', () => {
        render(<App />);

        const radio = screen.getByLabelText('detection');
        expect(radio).toBeChecked();
    });

    it('selects a task when the whole element is clicked', () => {
        const mockSetSelectedTask = rs.fn();
        render(<App setSelectedTask={mockSetSelectedTask} />);

        const segOption = screen.getByLabelText('Task option: Instance Segmentation');
        fireEvent.click(segOption);

        expect(mockSetSelectedTask).toHaveBeenCalledWith('instance_segmentation');
    });

    it('selects a task when the radio element is clicked', () => {
        const mockSetSelectedTask = rs.fn();
        render(<App setSelectedTask={mockSetSelectedTask} />);

        const classRadio = screen.getByLabelText('classification');
        fireEvent.click(classRadio);

        expect(mockSetSelectedTask).toHaveBeenCalledWith('classification');
    });
});
