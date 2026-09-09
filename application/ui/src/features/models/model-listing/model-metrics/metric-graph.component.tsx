// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import React, { useCallback, useMemo, useRef } from 'react';

import { ActionButton, DOMRefValue, Flex, useUnwrapDOMRef, View } from '@geti-ui/ui';
import { DownloadIcon } from '@geti-ui/ui/icons';
import { CartesianGrid, Line, LineChart, Tooltip, XAxis, YAxis } from 'recharts';

import { Box } from '../components/box/box.component';
import { downloadSvgAsImage } from './download-graph.utils';

export type MetricGraphPoint = {
    x: number;
    y: number;
};

type MetricGraphProps = {
    title: string;
    data?: MetricGraphPoint[];
    xAxisLabel?: string;
    yAxisLabel: string;
};

const X_AXIS_TICK_COUNT = 8;
const Y_AXIS_TICK_COUNT = 4;

export const MetricGraph = ({ title, data, xAxisLabel, yAxisLabel }: MetricGraphProps) => {
    const graphRef = useRef<DOMRefValue<HTMLDivElement>>(null);
    const unwrappedGraphRef = useUnwrapDOMRef(graphRef);

    const chartData = useMemo(() => {
        if (!data || data.length === 0) return [];
        // Ensure the line starts from zero epoch (x=0)
        if (data[0].x > 0) {
            return [{ x: 0, y: 0 }, ...data];
        }
        return data;
    }, [data]);

    const handleDownload = useCallback(() => {
        if (!unwrappedGraphRef.current) return;

        const svgElement = unwrappedGraphRef.current.querySelector('svg');
        if (!svgElement) return;

        void downloadSvgAsImage(svgElement, title);
    }, [title, unwrappedGraphRef]);

    return (
        <Flex flex={1} direction={'column'} minWidth={'size-5000'}>
            <Box
                title={title}
                actions={
                    <ActionButton isQuiet onPress={handleDownload} aria-label={`Download ${title} graph`}>
                        <DownloadIcon />
                    </ActionButton>
                }
                content={
                    <View ref={graphRef} backgroundColor={'gray-50'} minHeight={'size-3800'}>
                        <LineChart
                            responsive
                            width={'100%'}
                            style={{ aspectRatio: 1.6 }}
                            data={chartData}
                            margin={{ top: 35, bottom: 45, left: 35, right: 35 }}
                        >
                            <CartesianGrid />
                            <XAxis
                                dataKey='x'
                                type='number'
                                domain={[0, 'auto']}
                                allowDecimals={false}
                                label={{ value: xAxisLabel ?? 'x', position: 'bottom', fill: '#666', offset: 20 }}
                                tickCount={X_AXIS_TICK_COUNT}
                                tickMargin={12}
                            />
                            <YAxis
                                label={{ value: yAxisLabel, angle: -90, position: 'center', dx: -38, fill: '#666' }}
                                tickCount={Y_AXIS_TICK_COUNT}
                                tickMargin={12}
                                tickFormatter={(value) =>
                                    Number.isInteger(value) ? String(value) : Number(value).toFixed(4)
                                }
                            />
                            <Tooltip
                                contentStyle={{ backgroundColor: '#fff', border: '1px solid #ccc' }}
                                labelStyle={{ color: '#333' }}
                                formatter={(value: number | string | ReadonlyArray<string | number> | undefined) => [
                                    Number.isInteger(Number(value)) ? Number(value) : Number(value).toFixed(4),
                                    yAxisLabel,
                                ]}
                                labelFormatter={(label: React.ReactNode) => `${xAxisLabel ?? 'x'}: ${label}`}
                            />
                            <Line
                                type='linear'
                                dataKey='y'
                                name={yAxisLabel}
                                stroke='var(--energy-blue)'
                                strokeWidth={2}
                                dot={false}
                            />
                        </LineChart>
                    </View>
                }
            />
        </Flex>
    );
};
