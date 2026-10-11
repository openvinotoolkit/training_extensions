// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { View } from '@geti-ui/ui';

type BottomProgressBarProps = {
    progress: number;
    color?: string;
    children: ReactNode;
};

export const BottomProgressBar = ({ progress, color, children }: BottomProgressBarProps) => {
    return (
        <View position='relative'>
            {children}
            <View
                position='absolute'
                bottom={0}
                left={0}
                height='size-50'
                width={`${progress}%`}
                backgroundColor={color ? undefined : 'informative'}
                UNSAFE_style={color ? { backgroundColor: color } : undefined}
            />
        </View>
    );
};
