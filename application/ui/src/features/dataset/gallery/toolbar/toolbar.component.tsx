// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Dispatch, SetStateAction, Suspense, useMemo } from 'react';

import type { Media } from '@/api/types';
import {
    ActionButton,
    Button,
    ButtonGroup,
    Checkbox,
    dimensionValue,
    Divider,
    Flex,
    Heading,
    MediaViewModes,
    ViewModes,
} from '@geti-ui/ui';
import { SortDown, SortUp } from '@geti-ui/ui/icons';
import { useDatasetFiltersSearchParams } from 'hooks/use-dataset-filters-search-params.hook';
import { isString } from 'lodash-es';

import { FEATURE_FLAGS } from '../../../../constants/feature-flags';
import { isImage } from '../../../../shared/media-item-utils';
import { TrainModel } from '../../../models/train-model/train-model.component';
import { ImportExport } from '../../import-export/import-export.component';
import { useSelectedData } from '../../providers/selected-data-provider.component';
import { DeleteMediaItem } from '../delete-media-item/delete-media-item.component';
import { useSelectDatasetItem } from '../hooks/use-select-dataset-item.hook';
import { AssignLabel } from './assign-label.component';
import { DatasetStatistics } from './dataset-statistics/dataset-statistics.component';
import { useDatasetViewsQuery } from './dataset-view-selector/api/use-dataset-views';
import { AssignToExistingView } from './dataset-view-selector/assign-to-existing-view/assign-to-existing-view.component';
import { DatasetViewSelector } from './dataset-view-selector/dataset-view-selector.component';
import { SaveDatasetView } from './dataset-view-selector/save-dataset-view/save-dataset-view.component';
import { UnassignMediaFromView } from './dataset-view-selector/unassign-media-from-view/unassign-media-from-view.component';
import { MediaFiltering } from './media-filtering/media-filtering.component';
import { MediaUpload } from './media-upload.component';
import { TotalItems } from './total-items.component';
import { toggleMultipleSelection } from './util';

type ToolbarProps = {
    items: Media[];
    viewMode: ViewModes;
    setViewMode: Dispatch<SetStateAction<ViewModes>>;
};

type AnnotateButtonProps = {
    isDisabled?: boolean;
    onClick?: () => void;
};

const AnnotateButton = ({ isDisabled, onClick }: AnnotateButtonProps) => {
    return (
        <Button margin={0} variant={'primary'} onPress={onClick} isDisabled={isDisabled}>
            Annotate
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
    const { sortDirection, setSortDirection } = useDatasetFiltersSearchParams();

    if (sortDirection === 'asc') {
        return (
            <ActionButton isQuiet onPress={() => setSortDirection('desc')}>
                Oldest first <SortUp />
            </ActionButton>
        );
    }

    return (
        <ActionButton isQuiet onPress={() => setSortDirection('asc')}>
            Newest first <SortDown />
        </ActionButton>
    );
};

export const Toolbar = ({ items, viewMode, setViewMode }: ToolbarProps) => {
    const { onSelectedMediaItemChange } = useSelectDatasetItem();
    const { selectedKeys, setSelectedKeys, toggleSelectedKeys } = useSelectedData();

    const selectedMediaItems = selectedKeys instanceof Set ? selectedKeys : null;

    const totalSelectedElements = selectedMediaItems?.size ?? 0;
    const hasSelectedElements = totalSelectedElements > 0;

    const handleToggleManyItemSelection = () => {
        const images = items.map((item) => String(item.id));
        setSelectedKeys(toggleMultipleSelection(images));
    };

    const selectedImagesIds = useMemo(() => {
        if (selectedMediaItems === null) return [];

        return Array.from(selectedMediaItems)
            .filter((itemId) => items.some((item) => itemId === item.id && isImage(item)))
            .filter((itemId) => isString(itemId));
    }, [selectedMediaItems, items]);

    const resetSelectedMediaIds = () => {
        setSelectedKeys(new Set());
    };

    const noMediaSelected = selectedMediaItems?.size === 0;
    const selectedMediaItemsIds = Array.from(selectedMediaItems ?? []) as string[];

    return (
        <Flex direction={'column'} gridArea={'toolbar'} gap={'size-200'} marginBottom={'size-200'}>
            <Flex alignItems={'center'} justifyContent={'space-between'}>
                <Flex alignItems={'center'} gap={'size-200'}>
                    <Heading margin={0}>Dataset</Heading>

                    {FEATURE_FLAGS.DATASET_VIEWS && (
                        <Suspense fallback={null}>
                            <DatasetViews resetSelectedMediaIds={resetSelectedMediaIds} />
                        </Suspense>
                    )}
                </Flex>

                <ButtonGroup UNSAFE_style={{ gap: dimensionValue('size-125') }}>
                    {noMediaSelected && <ImportExport />}

                    <MediaUpload />

                    {noMediaSelected && <TrainModel />}

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
                        isSelected={hasSelectedElements && totalSelectedElements === items.length}
                    />

                    {!hasSelectedElements && <SortMediaByUploadDate />}

                    {hasSelectedElements && (
                        <>
                            <AssignLabel selectedImagesIds={selectedImagesIds} />
                            <DeleteMediaItem
                                itemsIds={Array.from(selectedKeys) as string[]}
                                onDeleted={toggleSelectedKeys}
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

                            <MediaViewModes
                                viewMode={viewMode}
                                setViewMode={setViewMode}
                                items={[ViewModes.LARGE, ViewModes.MEDIUM, ViewModes.SMALL]}
                            />
                        </>
                    )}
                </Flex>
            </Flex>

            <Divider size='S' />
        </Flex>
    );
};
