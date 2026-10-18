// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Suspense } from 'react';

import { $api } from '@/api';
import { GetiLogoLink } from '@/components/geti-logo-link/geti-logo-link.component';
import { HelpMenu } from '@/components/help-menu/help-menu.component';
import { LanguagesSelector } from '@/components/languages-selector/languages-selector.component';
import { useTranslation } from '@/i18n';
import { Divider, Flex, Grid, Item, Loading, TabList, Tabs, View } from '@geti-ui/ui';
import { usePrefetchQuery } from '@tanstack/react-query';
import { usePrefetchPipeline } from 'hooks/api/pipeline.hook';
import { useProject } from 'hooks/api/project.hook';
import { Outlet, useLocation } from 'react-router';

import { paths } from './constants/paths';
import { usePrefetchSinksQuery } from './features/inference/sinks/api/use-sinks-query';
import { usePrefetchSourcesQuery } from './features/inference/sources/api/use-sources';
import { JobsButton } from './features/jobs/jobs-button.component';
import { ProjectsListPanel } from './features/project/panel/projects-list-panel.component';
import { useProjectIdentifier } from './hooks/use-project-identifier.hook';

import classes from './layout.module.scss';

const Header = () => {
    const projectId = useProjectIdentifier();
    const { t } = useTranslation();

    return (
        <View backgroundColor={'gray-200'} gridArea={'header'}>
            <Grid
                height='100%'
                gap={'size-300'}
                marginStart={'size-300'}
                columns={['auto', '2fr', 'auto', 'fit-content(var(--spectrum-global-dimension-size-3400))']}
                rows={'1fr'}
                alignItems={'center'}
            >
                <View paddingEnd={'size-200'}>
                    <GetiLogoLink />
                </View>

                <TabList height={'100%'} UNSAFE_className={classes.tabList}>
                    <Item
                        textValue={t('navigation.datasetDescription')}
                        key={'dataset'}
                        href={paths.project.dataset.index({ projectId })}
                    >
                        {t('common.labels.dataset')}
                    </Item>
                    <Item
                        textValue={t('navigation.modelsDescription')}
                        key={'models'}
                        href={paths.project.models({ projectId })}
                    >
                        {t('navigation.models')}
                    </Item>
                    <Item
                        textValue={t('navigation.inferenceDescription')}
                        key={'inference'}
                        href={paths.project.inference({ projectId })}
                    >
                        {t('navigation.inference')}
                    </Item>
                </TabList>

                <Flex gap={'size-200'} alignItems={'center'}>
                    <Suspense fallback={<Loading />}>
                        <ProjectsListPanel />
                    </Suspense>

                    <Divider orientation={'vertical'} size={'S'} />

                    <Suspense fallback={<Loading size='S' mode='inline' />}>
                        <JobsButton />
                    </Suspense>

                    <LanguagesSelector />
                    <HelpMenu />
                </Flex>
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
