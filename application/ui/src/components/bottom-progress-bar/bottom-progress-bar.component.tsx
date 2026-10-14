// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { View } from '@geti-ui/ui';

type BottomProgressBarProps = {
    progress: number;
    color?: string;
    ariaLabel?: string;
    children: ReactNode;
};

export const BottomProgressBar = ({ progress, color, ariaLabel, children }: BottomProgressBarProps) => {
    return (
        <View position='relative'>
            {children}
            <div
                role={ariaLabel ? 'progressbar' : undefined}
                aria-label={ariaLabel}
                aria-valuemin={ariaLabel ? 0 : undefined}
                aria-valuemax={ariaLabel ? 100 : undefined}
                aria-valuenow={ariaLabel ? progress : undefined}
                style={{ position: 'absolute', bottom: 0, left: 0, width: `${progress}%` }}
            >
                <View
                    height='size-50'
                    width='100%'
                    backgroundColor={color ? undefined : 'informative'}
                    UNSAFE_style={color ? { backgroundColor: color } : undefined}
                />
            </div>
        </View>
    );
};
