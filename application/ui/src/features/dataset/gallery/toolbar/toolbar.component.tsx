// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Dispatch, ReactNode, SetStateAction, Suspense, useState } from 'react';

import type { Media } from '@/api/types';
import { GalleryViewModeMenu } from '@/components/gallery-view-mode-menu/gallery-view-mode-menu.component';
import { useTranslation } from '@/i18n';
import {
    ActionButton,
    Button,
    ButtonGroup,
    Checkbox,
    dimensionValue,
    Divider,
    Flex,
    Heading,
    ViewModes,
} from '@geti-ui/ui';
import { SortDown, SortUp } from '@geti-ui/ui/icons';
import { useDatasetViewsQuery } from 'hooks/api/dataset-views.hook';
import { useDatasetFiltersSearchParams } from 'hooks/use-dataset-filters-search-params.hook';
import { useDatasetMediaWithReviewStatus } from 'hooks/use-dataset-media-with-review-status.hook';
import { useSelectAllDatasetMedia } from 'hooks/use-select-all-dataset-media.hook';

import { FEATURE_FLAGS } from '../../../../constants/feature-flags';
import { isImage } from '../../../../shared/media-item-utils';
import { ImportExport } from '../../import-export/import-export.component';
import { useSelectedData } from '../../providers/selected-data-provider.component';
import { DeleteMediaItem } from '../delete-media-item/delete-media-item.component';
import { useSelectDatasetItem } from '../hooks/use-select-dataset-item.hook';
import { AssignLabel } from './assign-label.component';
import { DatasetStatistics } from './dataset-statistics/dataset-statistics.component';
import { AssignToExistingView } from './dataset-view-selector/assign-to-existing-view/assign-to-existing-view.component';
import { DatasetViewSelector } from './dataset-view-selector/dataset-view-selector.component';
import { SaveDatasetView } from './dataset-view-selector/save-dataset-view/save-dataset-view.component';
import { UnassignMediaFromView } from './dataset-view-selector/unassign-media-from-view/unassign-media-from-view.component';
import { MediaFiltering } from './media-filtering/media-filtering.component';
import { MediaUpload } from './media-upload.component';
import { TotalItems } from './total-items.component';

type ToolbarProps = {
    items: Media[];
    viewMode: ViewModes;
    setViewMode: Dispatch<SetStateAction<ViewModes>>;
    trainModel: ReactNode;
};

type AnnotateButtonProps = {
    isDisabled?: boolean;
    onClick?: () => void;
};

const AnnotateButton = ({ isDisabled, onClick }: AnnotateButtonProps) => {
    const { t } = useTranslation();

    return (
        <Button margin={0} variant={'primary'} onPress={onClick} isDisabled={isDisabled}>
            {t('common.actions.annotate')}
        </Button>
    );
};

type DatasetViewsProps = {
    resetSelectedMediaIds: () => void;
};

const DatasetViews = ({ resetSelectedMediaIds }: DatasetViewsProps) => {
    const { data: datasetViews } = useDatasetViewsQuery();

    return (
        <>
            <Divider orientation={'vertical'} size={'S'} />
            <DatasetViewSelector datasetViews={datasetViews} resetSelectedMediaIds={resetSelectedMediaIds} />
        </>
    );
};

type DatasetViewActionsProps = {
    selectedMediaIds: string[];
    resetSelectedMediaIds: () => void;
};

const DatasetViewActions = ({ selectedMediaIds, resetSelectedMediaIds }: DatasetViewActionsProps) => {
    const { data: datasetViews } = useDatasetViewsQuery();

    return (
        <>
            <SaveDatasetView
                selectedMediaIds={selectedMediaIds}
                datasetViews={datasetViews}
                resetSelectedMediaIds={resetSelectedMediaIds}
            />
            <AssignToExistingView
                datasetViews={datasetViews}
                selectedMediaIds={selectedMediaIds}
                resetSelectedMediaIds={resetSelectedMediaIds}
            />
        </>
    );
};

const SortMediaByUploadDate = () => {
    const { t } = useTranslation();
    const { sortDirection, setSortDirection } = useDatasetFiltersSearchParams();

    if (sortDirection === 'asc') {
        return (
            <ActionButton isQuiet onPress={() => setSortDirection('desc')}>
                {t('dataset.gallery.sortOldestFirst')} <SortUp />
            </ActionButton>
        );
    }

    return (
        <ActionButton isQuiet onPress={() => setSortDirection('asc')}>
            {t('dataset.gallery.sortNewestFirst')} <SortDown />
        </ActionButton>
    );
};

export const Toolbar = ({ items, viewMode, setViewMode, trainModel }: ToolbarProps) => {
    const { t } = useTranslation();
    const { selectedMediaItem, onSelectedMediaItemChange } = useSelectDatasetItem();
    const { selectedKeys, setSelectedKeys, toggleSelectedKeys } = useSelectedData();
    const { totalCount } = useDatasetMediaWithReviewStatus();
    const selectAllMedia = useSelectAllDatasetMedia();

    // Which of the ids resolved by "select all" are images, for the classification label actions.
    const [selectAllImageIds, setSelectAllImageIds] = useState<string[]>([]);

    const totalSelectedElements = selectedKeys.size;
    const hasSelectedElements = totalSelectedElements > 0;
    const allElementsSelected = totalCount > 0 && totalSelectedElements === totalCount;

    const handleToggleManyItemSelection = () => {
        if (allElementsSelected) {
            setSelectedKeys(new Set());

            return;
        }

        selectAllMedia.mutate(undefined, {
            onSuccess: (result) => {
                if (result === null) {
                    return;
                }

                setSelectAllImageIds(result.imageIds);
                setSelectedKeys(new Set(result.mediaIds));
            },
        });
    };

    // The gallery only holds the pages it has loaded, so ids resolved by "select all" are the
    // only way to tell whether an unloaded selected item is an image.
    const imageIds = new Set(selectAllImageIds);
    items.filter(isImage).forEach((item) => imageIds.add(String(item.id)));

    const selectedImagesIds = Array.from(selectedKeys).filter((itemId) => imageIds.has(itemId));

    const resetSelectedMediaIds = () => {
        setSelectedKeys(new Set());
    };

    const noMediaSelected = selectedKeys.size === 0;
    const selectedMediaItemsIds = Array.from(selectedKeys);

    return (
        <Flex direction={'column'} gridArea={'toolbar'} gap={'size-200'} marginBottom={'size-200'}>
            <Flex alignItems={'center'} justifyContent={'space-between'}>
                <Flex alignItems={'center'} gap={'size-200'}>
                    <Heading margin={0}>{t('common.labels.dataset')}</Heading>

                    {FEATURE_FLAGS.DATASET_VIEWS && (
                        <Suspense fallback={null}>
                            <DatasetViews resetSelectedMediaIds={resetSelectedMediaIds} />
                        </Suspense>
                    )}
                </Flex>

                <ButtonGroup UNSAFE_style={{ gap: dimensionValue('size-125') }}>
                    {noMediaSelected && <ImportExport />}

                    <MediaUpload />

                    {noMediaSelected && trainModel}

                    {noMediaSelected && (
                        <AnnotateButton
                            isDisabled={items.at(0) === undefined}
                            onClick={items.at(0) === undefined ? undefined : () => onSelectedMediaItemChange(items[0])}
                        />
                    )}
                </ButtonGroup>
            </Flex>

            <Divider size='S' />

            <Flex direction={'row'} alignItems={'center'} justifyContent={'space-between'}>
                <Flex
                    gap={'size-100'}
                    height={'size-400'}
                    direction={'row'}
                    alignItems={'center'}
                    justifyContent={'space-between'}
                >
                    <Checkbox
                        aria-label={'select all'}
                        onChange={handleToggleManyItemSelection}
                        isSelected={allElementsSelected}
                        isIndeterminate={hasSelectedElements && !allElementsSelected}
                        isDisabled={totalCount === 0 || selectAllMedia.isPending}
                    />

                    {!hasSelectedElements && <SortMediaByUploadDate />}

                    {hasSelectedElements && (
                        <>
                            <AssignLabel selectedImagesIds={selectedImagesIds} />
                            <DeleteMediaItem
                                itemsIds={selectedMediaItemsIds}
                                onDeleted={toggleSelectedKeys}
                                isHotkeyEnabled={selectedMediaItem === null}
                            />
                            {FEATURE_FLAGS.DATASET_VIEWS && (
                                <>
                                    <Suspense fallback={null}>
                                        <DatasetViewActions
                                            selectedMediaIds={selectedMediaItemsIds}
                                            resetSelectedMediaIds={resetSelectedMediaIds}
                                        />
                                    </Suspense>
                                    <UnassignMediaFromView
                                        selectedMediaIds={selectedMediaItemsIds}
                                        resetSelectedMediaIds={resetSelectedMediaIds}
                                    />
                                </>
                            )}
                        </>
                    )}
                </Flex>

                <Flex gap={'size-200'} alignItems={'center'}>
                    <TotalItems totalSelectedElements={totalSelectedElements} />

                    {noMediaSelected && (
                        <>
                            <MediaFiltering />

                            <DatasetStatistics />

                            <GalleryViewModeMenu viewMode={viewMode} setViewMode={setViewMode} />
                        </>
                    )}
                </Flex>
            </Flex>

            <Divider size='S' />
        </Flex>
    );
};
