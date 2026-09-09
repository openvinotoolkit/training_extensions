// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useRef, useState } from 'react';

import { Content, Heading, IllustratedMessage, View } from '@geti-ui/ui';
import { usePipelineMetrics } from 'hooks/api/pipeline.hook';
import { CartesianGrid, Label, Line, LineChart, ReferenceLine, Tooltip, XAxis, YAxis } from 'recharts';

type DataPoint = {
    name: string;
    value: number;
};

const MAX_DATA_POINTS = 60; // Keep last 60 data points

const useMetricsData = () => {
    const [latencyData, setLatencyData] = useState<DataPoint[]>([]);
    const [throughputData, setThroughputData] = useState<DataPoint[]>([]);
    const counterRef = useRef(0);

    const { data: metrics } = usePipelineMetrics();

    useEffect(() => {
        if (!metrics) return;

        const dataPointName = `${counterRef.current++}`;

        setLatencyData((prev) => {
            const newData = [
                ...prev,
                {
                    name: dataPointName,
                    value: metrics.inference.latency.avg_ms ?? 0,
                },
            ];

            // Keep only last MAX_DATA_POINTS
            return newData.slice(-MAX_DATA_POINTS);
        });

        setThroughputData((prev) => {
            const newData = [
                ...prev,
                {
                    name: dataPointName,
                    value: metrics.inference.throughput.avg_requests_per_second ?? 0,
                },
            ];
            return newData.slice(-MAX_DATA_POINTS);
        });
    }, [metrics]);

    return { latencyData, throughputData, metrics };
};

const AXIS_LABEL_STYLE = {
    textAnchor: 'middle',
    fill: 'var(--spectrum-global-color-gray-900)',
    fontSize: '10px',
} as const;

const formatValue = (value: unknown) => {
    const raw = Array.isArray(value) ? value[0] : value;
    const num = typeof raw === 'number' ? raw : Number(raw);

    if (!Number.isFinite(num)) {
        return String(raw ?? '');
    }

    return num > 10 ? num.toFixed(0) : num.toFixed(2);
};

const Graph = ({ label, data }: { label: string; data: DataPoint[] }) => {
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
                dataKey='name'
                tickLine={false}
                tickMargin={8}
            >
                <Label value='samples' position='insideBottom' offset={-14} style={AXIS_LABEL_STYLE} />
            </XAxis>
            <YAxis
                tickLine={false}
                stroke='var(--spectrum-global-color-gray-900)'
                dataKey='value'
                tickFormatter={formatValue}
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
                labelFormatter={(name) => `Sample ${name}`}
                formatter={(value) => formatValue(Number(value))}
            />
            {data.length > 0 && (
                <ReferenceLine x={data[0].name} stroke='var(--spectrum-global-color-gray-600)' strokeWidth={2} />
            )}
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
    const { latencyData, throughputData, metrics } = useMetricsData();

    const hasData = latencyData.length > 0 || throughputData.length > 0;

    return (
        <View height={'100%'} UNSAFE_style={{ overflow: 'hidden auto' }}>
            {!hasData && !metrics ? (
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
                        <Graph label='requests/sec' data={throughputData} />
                    </View>
                    <View>
                        <Heading level={4} marginBottom={'size-300'}>
                            Latency
                        </Heading>
                        <Graph label='ms' data={latencyData} />
                    </View>
                </>
            )}
        </View>
    );
};
