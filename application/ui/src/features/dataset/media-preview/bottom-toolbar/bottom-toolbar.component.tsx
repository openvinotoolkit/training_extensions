// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { DatasetSubset, Media } from '@/api/types';
import { useTranslation } from '@/i18n';
import { Flex, Grid, Item, Key, Picker, Tag, Text } from '@geti-ui/ui';
import { Accept, Search } from '@geti-ui/ui/icons';
import { clsx } from 'clsx';

import { Hotkeys } from '../primary-toolbar/hotkeys/hotkeys.component';
import { Settings } from '../primary-toolbar/settings/settings.component';
import { ToggleFocus } from '../primary-toolbar/toggle-focus.component';
import { ZoomFitScreen } from '../primary-toolbar/zoom/zoom-fit-screen.component';
import { ZoomSelector } from '../primary-toolbar/zoom/zoom-selector.component';
import { Toolbar } from '../toolbar-container/toolbar-container.component';

import classes from './bottom-toolbar.module.scss';

type BottomToolbarProps = {
    mediaItem: Media;
    isUserReviewed?: boolean;
    subset: DatasetSubset;
    onSubsetChange?: (key: Key | null) => void;
    hideHotkeys?: boolean;
    isReadOnlySubset: boolean;
    hasAnnotationStatus?: boolean;
};

export const BottomToolbar = ({
    hideHotkeys,
    mediaItem,
    isUserReviewed,
    subset,
    onSubsetChange,
    isReadOnlySubset,
    hasAnnotationStatus = true,
}: BottomToolbarProps) => {
    const { t } = useTranslation();
    const fileName = `${mediaItem.name}.${mediaItem.format} (${mediaItem.width} x ${mediaItem.height} px)`;

    return (
        <Flex justifyContent={'end'}>
            <Toolbar.Container>
                <Grid autoFlow={'column'} autoColumns={'max-content'} gap={'size-50'}>
                    {!hideHotkeys && (
                        <Toolbar.Section>
                            <Hotkeys />
                        </Toolbar.Section>
                    )}

                    <Toolbar.Section>
                        <Flex gap={'size-100'} alignItems={'center'} height={'100%'}>
                            <Text UNSAFE_className={classes.filename}>{fileName}</Text>
                            {hasAnnotationStatus && (
                                <Tag
                                    className={clsx({
                                        [classes.reviewed]: isUserReviewed,
                                        [classes.forReview]: !isUserReviewed,
                                    })}
                                    prefix={isUserReviewed ? <Accept /> : <Search />}
                                    text={isUserReviewed ? t('dataset.review.reviewed') : t('dataset.review.forReview')}
                                />
                            )}

                            {isReadOnlySubset ? (
                                <Tag
                                    withDot={false}
                                    text={t(`dataset.filters.subsetOptions.${subset}`)}
                                    id={'selected-subset-badge'}
                                />
                            ) : (
                                <Picker
                                    selectedKey={subset}
                                    placeholder={t('dataset.review.selectSubsetPlaceholder')}
                                    aria-label={'Select subset'}
                                    onSelectionChange={onSubsetChange}
                                >
                                    <Item key={'unassigned'}>{t('dataset.filters.subsetOptions.unassigned')}</Item>
                                    <Item key={'validation'}>{t('dataset.filters.subsetOptions.validation')}</Item>
                                    <Item key={'testing'}>{t('dataset.filters.subsetOptions.testing')}</Item>
                                    <Item key={'training'}>{t('dataset.filters.subsetOptions.training')}</Item>
                                </Picker>
                            )}
                        </Flex>
                    </Toolbar.Section>

                    <Toolbar.Section>
                        <Flex alignItems={'center'}>
                            <Settings />

                            <ZoomSelector />

                            <ToggleFocus />

                            <ZoomFitScreen />
                        </Flex>
                    </Toolbar.Section>
                </Grid>
            </Toolbar.Container>
        </Flex>
    );
};
