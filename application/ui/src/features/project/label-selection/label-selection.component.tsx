// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Dispatch, SetStateAction } from 'react';

import type { Label, TaskType } from '@/api/types';
import { toast } from '@/components/toast/toast.component';
import { Flex } from '@geti-ui/ui';

import { CreateLabel } from './create-label/create-label.component';
import { LabelTag } from './label-tag/label-tag.component';

type LabelSelectionProps = {
    labels: Label[];
    setLabels: Dispatch<SetStateAction<Label[]>>;
    taskType: TaskType;
};

export const LabelSelection = ({ labels, setLabels, taskType }: LabelSelectionProps) => {
    const handleDeleteItem = (id: string) => {
        const newLabels = labels.filter((label) => label.id !== id);
        setLabels(newLabels);

        if (newLabels.length === 0) {
            toast({ type: 'info', message: 'At least one object is required' });
        }
    };

    const handleAddItem = (label: Label) => {
        setLabels([...labels, label]);
    };

    return (
        <Flex
            direction={'column'}
            alignItems={'center'}
            height={'100%'}
            width={'100%'}
            gap={'size-300'}
            UNSAFE_style={{ overflow: 'auto' }}
        >
            <CreateLabel onCreate={handleAddItem} labels={labels} taskType={taskType} />
            <Flex gap={'size-100'} width={'100%'} wrap={'wrap'} justifyContent={'center'} alignItems={'center'}>
                {labels.map((label) => (
                    <LabelTag key={label.id} label={label} onDelete={handleDeleteItem} />
                ))}
            </Flex>
        </Flex>
    );
};
