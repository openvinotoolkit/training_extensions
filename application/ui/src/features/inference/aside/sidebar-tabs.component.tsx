// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode, useState } from 'react';

import { useTranslation } from '@/i18n';
import { Flex, Grid, Heading, ToggleButton, Tooltip, TooltipTrigger, View } from '@geti-ui/ui';
import { Gear, GraphChart } from '@geti-ui/ui/icons';

import { ReactComponent as PipelineIcon } from '../../../assets/icons/pipeline.svg';
import { DataCollection } from './data-collection.component';
import { Graphs } from './graphs.component';
import { PipelineConfiguration } from './pipeline-configuration.component';

import styles from './sidebar-tabs.module.scss';

type Tab = {
    id: string;
    label: string;
    ariaLabel: string;
    icon: ReactNode;
    content: ReactNode;
};

type TabProps = {
    tabs: Tab[];
    selectedTab: string;
};

const SidebarTabs = ({ tabs, selectedTab }: TabProps) => {
    const [tab, setTab] = useState<string | null>(selectedTab);

    const isExpanded = tab !== null;
    const gridTemplateColumns = isExpanded ? ['clamp(size-4600, 30vw, 40rem)', 'size-600'] : ['0px', 'size-600'];

    const content = tabs.find(({ id }) => id === tab)?.content;

    const handleSetTab = (id: string) => {
        setTab((prev) => (prev === id ? null : id));
    };

    return (
        <Grid
            gridArea={'aside'}
            UNSAFE_className={styles.container}
            columns={gridTemplateColumns}
            data-expanded={isExpanded}
            minHeight={0}
        >
            <View
                gridColumn={'1/2'}
                UNSAFE_className={styles.sidebarContent}
                backgroundColor={'gray-100'}
                paddingY={'size-400'}
                paddingX={'size-500'}
                aria-hidden={!isExpanded || undefined}
            >
                {isExpanded && (
                    <>
                        <Flex alignItems='center' gap={'size-100'} marginBottom={'size-300'}>
                            <Heading level={2}>{tabs.find((item) => item.id === tab)?.label}</Heading>
                        </Flex>
                        <Flex direction={'column'} flex={1} UNSAFE_style={{ overflow: 'hidden auto' }}>
                            {content}
                        </Flex>
                    </>
                )}
            </View>
            <View gridColumn={'2/3'} backgroundColor={'gray-200'} padding={'size-100'}>
                <Flex direction={'column'} height={'100%'} alignItems={'center'} gap={'size-100'}>
                    {tabs.map(({ id, label, ariaLabel, icon }) => (
                        <TooltipTrigger key={id} placement={'left'}>
                            <ToggleButton
                                isQuiet
                                isSelected={id === tab}
                                onChange={() => handleSetTab(id)}
                                UNSAFE_className={styles.toggleButton}
                                aria-label={`Toggle ${ariaLabel} tab`}
                            >
                                {icon}
                            </ToggleButton>
                            <Tooltip>{label}</Tooltip>
                        </TooltipTrigger>
                    ))}
                </Flex>
            </View>
        </Grid>
    );
};

export const Sidebar = () => {
    const { t } = useTranslation();

    const TABS: Tab[] = [
        {
            id: 'configuration',
            label: t('inference.pipeline.configuration.sidebarLabel'),
            ariaLabel: 'Pipeline configuration',
            icon: <PipelineIcon />,
            content: <PipelineConfiguration />,
        },
        {
            id: 'dataCollection',
            label: t('inference.collection.sidebarLabel'),
            ariaLabel: 'Data collection policy',
            icon: <Gear />,
            content: <DataCollection />,
        },
        {
            id: 'metrics',
            label: t('inference.metrics.sidebarLabel'),
            ariaLabel: 'Pipeline metrics',
            icon: <GraphChart />,
            content: <Graphs />,
        },
    ];

    return <SidebarTabs tabs={TABS} selectedTab={TABS[0].id} />;
};
