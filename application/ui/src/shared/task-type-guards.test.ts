// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { TaskType } from '@/api/types';
import { getMockedProject } from 'mocks/mock-project';

import {
    isClassificationTask,
    isDetectionTask,
    isMultiLabelClassificationTask,
    isSegmentationTask,
} from './task-type-guards';

describe('isClassificationTask', () => {
    it('returns true for "classification"', () => {
        expect(isClassificationTask('classification')).toBe(true);
    });

    it.each(['detection', 'instance_segmentation'] satisfies TaskType[])('returns false for "%s"', (taskType) => {
        expect(isClassificationTask(taskType)).toBe(false);
    });

    it('returns false for null', () => {
        expect(isClassificationTask(null)).toBe(false);
    });
});

describe('isDetectionTask', () => {
    it('returns true for "detection"', () => {
        expect(isDetectionTask('detection')).toBe(true);
    });

    it.each(['classification', 'instance_segmentation'] satisfies TaskType[])('returns false for "%s"', (taskType) => {
        expect(isDetectionTask(taskType)).toBe(false);
    });

    it('returns false for null', () => {
        expect(isDetectionTask(null)).toBe(false);
    });
});

describe('isSegmentationTask', () => {
    it('returns true for "instance_segmentation"', () => {
        expect(isSegmentationTask('instance_segmentation')).toBe(true);
    });

    it.each(['classification', 'detection'] satisfies TaskType[])('returns false for "%s"', (taskType) => {
        expect(isSegmentationTask(taskType)).toBe(false);
    });

    it('returns false for null', () => {
        expect(isSegmentationTask(null)).toBe(false);
    });
});

describe('isMultiLabelClassificationTask', () => {
    it('returns true when task is classification and exclusive_labels is false', () => {
        const task = getMockedProject({
            task: { task_type: 'classification', exclusive_labels: false, labels: [] },
        }).task;
        expect(isMultiLabelClassificationTask(task)).toBe(true);
    });

    it('returns false when task is classification but exclusive_labels is true', () => {
        const task = getMockedProject({
            task: { task_type: 'classification', exclusive_labels: true, labels: [] },
        }).task;
        expect(isMultiLabelClassificationTask(task)).toBe(false);
    });

    it('returns false when task is detection and exclusive_labels is false', () => {
        const task = getMockedProject({ task: { task_type: 'detection', exclusive_labels: false, labels: [] } }).task;
        expect(isMultiLabelClassificationTask(task)).toBe(false);
    });

    it('returns false when task is detection and exclusive_labels is true', () => {
        const task = getMockedProject({ task: { task_type: 'detection', exclusive_labels: true, labels: [] } }).task;
        expect(isMultiLabelClassificationTask(task)).toBe(false);
    });

    it('returns false when task is instance_segmentation and exclusive_labels is false', () => {
        const task = getMockedProject({
            task: { task_type: 'instance_segmentation', exclusive_labels: false, labels: [] },
        }).task;
        expect(isMultiLabelClassificationTask(task)).toBe(false);
    });

    it('returns false when task is instance_segmentation and exclusive_labels is true', () => {
        const task = getMockedProject({
            task: { task_type: 'instance_segmentation', exclusive_labels: true, labels: [] },
        }).task;
        expect(isMultiLabelClassificationTask(task)).toBe(false);
    });
});
