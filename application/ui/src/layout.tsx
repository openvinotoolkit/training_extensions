// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Suspense } from 'react';

import { $api } from '@/api';
import { ProjectsListPanel } from '@/components/project-panel/projects-list-panel.component';
import { Flex, Grid, Item, Loading, TabList, Tabs, Text, View } from '@geti-ui/ui';
import { usePrefetchQuery } from '@tanstack/react-query';
import { usePrefetchPipeline } from 'hooks/api/pipeline.hook';
import { useProject } from 'hooks/api/project.hook';
import { Link, Outlet, useLocation } from 'react-router-dom';

import getiLogo from './assets/icons/geti-logo.webp';
import { paths } from './constants/paths';
import { usePrefetchSinksQuery } from './features/inference/sinks/api/use-sinks-query';
import { usePrefetchSourcesQuery } from './features/inference/sources/api/use-sources';
import { useProjectIdentifier } from './hooks/use-project-identifier.hook';

import classes from './layout.module.scss';

const Header = () => {
    const projectId = useProjectIdentifier();

    return (
        <View backgroundColor={'gray-200'} gridArea={'header'}>
            <Grid
                height='100%'
                gap={'size-300'}
                marginStart={'size-300'}
                marginEnd={'size-200'}
                columns={['auto', '2fr', 'fit-content(var(--spectrum-global-dimension-size-3000))']}
                rows={'1fr'}
                alignItems={'center'}
            >
                <View paddingEnd={'size-200'}>
                    <Link to={paths.project.index({})} viewTransition>
                        <Flex alignItems='center' gap='size-50'>
                            <img src={getiLogo} alt={'Geti logo'} className={classes.logo} />
                            <Text UNSAFE_className={classes.logoText}>Geti™</Text>
                        </Flex>
                    </Link>
                </View>

                <TabList height={'100%'} UNSAFE_className={classes.tabList}>
                    <Item
                        textValue='Data collection page to visualise your media items'
                        key={'dataset'}
                        href={paths.project.dataset.index({ projectId })}
                    >
                        Dataset
                    </Item>
                    <Item
                        textValue='Models page to visualise your models'
                        key={'models'}
                        href={paths.project.models({ projectId })}
                    >
                        Models
                    </Item>
                    <Item
                        textValue='Inference page showing live inference on your project'
                        key={'inference'}
                        href={paths.project.inference({ projectId })}
                    >
                        Inference
                    </Item>
                </TabList>

                <Suspense fallback={<Loading />}>
                    <ProjectsListPanel />
                </Suspense>
            </Grid>
        </View>
    );
};

const getFirstPathSegment = (path: string): string => {
    return path.split('/').pop() || '';
};

const usePrefetchResources = () => {
    usePrefetchQuery($api.queryOptions('get', '/api/system/devices/inference'));
    usePrefetchPipeline();
    usePrefetchSourcesQuery();
    usePrefetchSinksQuery();
};

export const Layout = () => {
    const { pathname } = useLocation();
    // We want to check if the project exists before rendering the layout. If it doesn't, error boundary will catch it.
    useProject();
    usePrefetchResources();

    return (
        <Tabs aria-label='Header navigation' selectedKey={getFirstPathSegment(pathname)}>
            <Grid
                areas={['header', 'content']}
                rows={['size-800', 'minmax(0, 1fr)']}
                minHeight={'100vh'}
                maxHeight={'100vh'}
                height={'100%'}
            >
                <Header />

                <View backgroundColor={'gray-50'} gridArea={'content'} position={'relative'}>
                    <Suspense fallback={<Loading />}>
                        <Outlet />
                    </Suspense>
                </View>
            </Grid>
        </Tabs>
    );
};
