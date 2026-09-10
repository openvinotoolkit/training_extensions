// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState } from 'react';

import { Content, Heading, IllustratedMessage, View } from '@geti-ui/ui';
import dayjs from 'dayjs';
import { usePipelineMetrics } from 'hooks/api/pipeline.hook';
import { CartesianGrid, Label, Line, LineChart, Tooltip, XAxis, YAxis } from 'recharts';

type DataPoint = {
    timestamp: number;
    value: number;
};

const DISPLAYED_WINDOW_MS = 5 * 60 * 1000;

const formatTime = (timestamp: number) => dayjs(timestamp).format('HH:mm:ss');

const appendDataPoint = (points: DataPoint[], point: DataPoint): DataPoint[] =>
    [...points, point].filter(({ timestamp }) => timestamp > point.timestamp - DISPLAYED_WINDOW_MS);

const useMetricsData = () => {
    const [latencyData, setLatencyData] = useState<DataPoint[]>([]);
    const [throughputData, setThroughputData] = useState<DataPoint[]>([]);

    const { data: metrics } = usePipelineMetrics();

    useEffect(() => {
        if (!metrics) return;

        const timestamp = dayjs(metrics.time_window.end).valueOf();

        setLatencyData((prev) => appendDataPoint(prev, { timestamp, value: metrics.inference.latency.avg_ms ?? 0 }));
        setThroughputData((prev) =>
            appendDataPoint(prev, {
                timestamp,
                value: metrics.inference.throughput.avg_requests_per_second ?? 0,
            })
        );
    }, [metrics]);

    return { latencyData, throughputData };
};

const AXIS_LABEL_STYLE = {
    textAnchor: 'middle',
    fill: 'var(--spectrum-global-color-gray-900)',
    fontSize: '10px',
} as const;

type GraphProps = {
    label: string;
    data: DataPoint[];
    fractionDigits: number;
};

const Graph = ({ label, data, fractionDigits }: GraphProps) => {
    const end = data.at(-1)?.timestamp ?? Date.now();

    return (
        <LineChart
            responsive
            width={'100%'}
            style={{ aspectRatio: 1.6 }}
            data={data}
            margin={{ top: 5, right: 5, left: 5, bottom: 16 }}
        >
            <XAxis
                minTickGap={32}
                stroke='var(--spectrum-global-color-gray-800)'
                dataKey='timestamp'
                type='number'
                scale='time'
                domain={[end - DISPLAYED_WINDOW_MS, end]}
                tickFormatter={formatTime}
                tickLine={false}
                tickMargin={8}
            >
                <Label value='time' position='insideBottom' offset={-14} style={AXIS_LABEL_STYLE} />
            </XAxis>
            <YAxis
                tickLine={false}
                stroke='var(--spectrum-global-color-gray-900)'
                dataKey='value'
                tickFormatter={(value: number) => value.toFixed(0)}
            >
                <Label angle={-90} value={label} position='insideLeft' style={AXIS_LABEL_STYLE} />
            </YAxis>
            <CartesianGrid stroke='var(--spectrum-global-color-gray-400)' />
            <Tooltip
                contentStyle={{
                    backgroundColor: 'var(--spectrum-global-color-gray-100)',
                    border: '1px solid var(--spectrum-global-color-gray-400)',
                }}
                labelStyle={{ color: 'var(--spectrum-global-color-gray-900)' }}
                itemStyle={{ color: 'var(--spectrum-global-color-gray-900)' }}
                labelFormatter={(timestamp) => formatTime(Number(timestamp))}
                formatter={(value) => Number(value).toFixed(fractionDigits)}
            />
            <Line
                type='linear'
                dataKey='value'
                name={label}
                dot={false}
                stroke='var(--energy-blue)'
                isAnimationActive={false}
                strokeWidth='3'
            />
        </LineChart>
    );
};

export const Graphs = () => {
    const { latencyData, throughputData } = useMetricsData();

    return (
        <View height={'100%'} UNSAFE_style={{ overflow: 'hidden auto' }}>
            {latencyData.length === 0 ? (
                <IllustratedMessage>
                    <Heading>No statistics available</Heading>
                    <Content>
                        Pipeline metrics will show here once the pipeline starts running and processing data.
                    </Content>
                </IllustratedMessage>
            ) : (
                <>
                    <View>
                        <Heading level={4} marginBottom={'size-300'}>
                            Throughput
                        </Heading>
                        <Graph label='requests/sec' data={throughputData} fractionDigits={2} />
                    </View>
                    <View>
                        <Heading level={4} marginBottom={'size-300'}>
                            Latency
                        </Heading>
                        <Graph label='ms' data={latencyData} fractionDigits={1} />
                    </View>
                </>
            )}
        </View>
    );
};
