// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { Flex, Text, View } from '@geti-ui/ui';
import { Pie, PieChart, Sector } from 'recharts';

import classes from './dataset-statistics.module.scss';

const COLORS: Record<string, string> = {
    totalAnnotated: '#ff9f66',
    totalUnannotated: '#c55400',
};

type DatasetStatisticsProps = {
    label: string;
    totalMediaItems: number;
    totalAnnotatedItems: number;
};

export const DatasetStatistics = ({ label, totalMediaItems, totalAnnotatedItems }: DatasetStatisticsProps) => {
    const { t } = useTranslation();
    const totalUnannotatedItems = totalMediaItems - totalAnnotatedItems;
    const percentageAnnotated = totalMediaItems > 0 ? Math.round((totalAnnotatedItems / totalMediaItems) * 100) : 0;
    const percentageUnannotated = totalMediaItems > 0 ? Math.round((totalUnannotatedItems / totalMediaItems) * 100) : 0;

    return (
        <View backgroundColor='gray-75' padding='size-200' borderRadius='regular'>
            <Flex alignItems='start' justifyContent='center' gap='size-200'>
                <Flex
                    direction='column'
                    alignItems='end'
                    justifyContent='center'
                    UNSAFE_className={classes.unannotatedStats}
                >
                    <Text>{t('dataset.statistics.unannotated')}</Text>
                    <Text>{percentageUnannotated}%</Text>
                    <Text>
                        {totalUnannotatedItems} {label}
                    </Text>
                </Flex>

                <Flex
                    width='size-1600'
                    height='size-1600'
                    position='relative'
                    direction='column'
                    alignItems='center'
                    justifyContent='center'
                >
                    <PieChart width={134} height={134}>
                        <Pie
                            data={[
                                { name: 'totalAnnotated', value: totalAnnotatedItems },
                                { name: 'totalUnannotated', value: totalUnannotatedItems },
                            ]}
                            dataKey='value'
                            innerRadius={46}
                            outerRadius={58}
                            startAngle={90}
                            endAngle={-270}
                            shape={(props) => (
                                <Sector
                                    {...props}
                                    fill={COLORS[String(props.name)]}
                                    stroke='var(--spectrum-global-color-gray-75)'
                                />
                            )}
                        />
                    </PieChart>
                    <Flex direction='column' UNSAFE_className={classes.totalMedia}>
                        <Text UNSAFE_className={classes.totalMediaItems}>{totalMediaItems}</Text>
                        <Text UNSAFE_className={classes.mediaSubtitle}>{label}</Text>
                    </Flex>
                </Flex>

                <Flex
                    alignSelf='end'
                    direction='column'
                    alignItems='start'
                    justifyContent='center'
                    UNSAFE_className={classes.unannotatedStats}
                >
                    <Text>{t('dataset.statistics.annotated')}</Text>
                    <Text>{percentageAnnotated}%</Text>
                    <Text>
                        {totalAnnotatedItems} {label}
                    </Text>
                </Flex>
            </Flex>
        </View>
    );
};
