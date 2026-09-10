// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useMemo } from 'react';

import { useTranslation } from '@/i18n';
import { type ColorValue } from '@geti-ui/ui';

interface CircularProgressProps {
    percentage: number;
    size?: number;
    labelFontSize?: number;
    strokeWidth?: number;
    labelFontColor?: ColorValue;
    backStrokeColor?: ColorValue;
    color?: ColorValue;
    hasError?: boolean;
    checkMarkOnComplete?: boolean;
    checkMarkSize?: number;
    checkMarkColor?: ColorValue | string;
}

export const CircularProgress = ({
    percentage,
    size = 50,
    strokeWidth = 4,
    labelFontSize = 10,
    labelFontColor = 'gray-600',
    backStrokeColor = 'gray-100',
    color = 'blue-400',
    hasError = false,
    checkMarkSize = 50,
    checkMarkOnComplete = true,
    checkMarkColor = '--energy-blue-shade',
}: CircularProgressProps) => {
    const { t } = useTranslation();
    const progress = Math.floor(Math.max(0, Math.min(100, percentage)));

    const viewBox = useMemo<string>((): string => `0 0 ${size} ${size}`, [size]);
    const radius = useMemo<number>((): number => (size - strokeWidth) / 2, [size, strokeWidth]);
    const circumference = useMemo<number>((): number => radius * Math.PI * 2, [radius]);
    const dash = useMemo<number>((): number => (progress * circumference) / 100, [progress, circumference]);
    const getCheckMarkColor = useMemo<string>(
        (): string =>
            checkMarkColor.startsWith('--') ? `var(${checkMarkColor})` : `var(--spectrum-global-color-${color})`,
        [checkMarkColor, color]
    );

    return (
        <svg width={size} height={size} viewBox={viewBox} aria-label='progress circular loader'>
            <circle
                fill='none'
                stroke={`var(--spectrum-global-color-${backStrokeColor})`}
                cx={size / 2}
                cy={size / 2}
                r={radius}
                strokeWidth={`${strokeWidth}px`}
            />
            <circle
                fill='none'
                stroke={
                    checkMarkOnComplete
                        ? progress < 100
                            ? `var(--spectrum-global-color-${color})`
                            : getCheckMarkColor
                        : `var(--spectrum-global-color-${color})`
                }
                cx={size / 2}
                cy={size / 2}
                r={radius}
                strokeWidth={`${strokeWidth}px`}
                transform={`rotate(-90 ${size / 2} ${size / 2})`}
                strokeDasharray={`${[dash, circumference - dash]}`}
                strokeLinecap='square'
                style={{ transition: 'all 0.5s' }}
            />
            <text
                fill={
                    checkMarkOnComplete
                        ? progress < 100
                            ? `var(--spectrum-global-color-${labelFontColor})`
                            : getCheckMarkColor
                        : `var(--spectrum-global-color-${labelFontColor})`
                }
                fontSize={
                    checkMarkOnComplete
                        ? progress < 100
                            ? `${labelFontSize}px`
                            : `${checkMarkSize}px`
                        : `${labelFontSize}px`
                }
                dy={
                    checkMarkOnComplete
                        ? progress < 100
                            ? `${labelFontSize / 2}px`
                            : `${checkMarkSize / 2.5}px`
                        : `${labelFontSize / 2}px`
                }
                textAnchor='middle'
                x='50%'
                y='50%'
            >
                {hasError
                    ? t('dataset.import.progressUnavailable')
                    : checkMarkOnComplete
                      ? progress < 100
                          ? `${progress}%`
                          : '✓'
                      : `${progress}%`}
            </text>
        </svg>
    );
};
