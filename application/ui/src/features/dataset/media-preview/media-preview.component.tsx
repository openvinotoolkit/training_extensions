// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useMemo } from 'react';

import type { DatasetSubset, Media } from '@/api/types';
import { Content, Dialog, Grid, View } from '@geti-ui/ui';
import { useQueryClient } from '@tanstack/react-query';
import { useDatasetMediaWithReviewStatus } from 'hooks/use-dataset-media-with-review-status.hook';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import type { AnnotatorMode } from '../../../shared/annotator/annotator-mode';
import { ToolProvider } from '../../../shared/annotator/tool-provider.component';
import { isVideoFrame } from '../../../shared/media-item-utils';
import { getMediaPredictionsQueryKeyPrefix, useMediaPredictions } from '../../annotator/api/use-media-predictions';
import { PredictionsSetupProvider, usePredictionSetup } from '../../annotator/predictions-setup-provider.component';
import {
    SelectedMediaItemProvider,
    useSelectedMediaItem,
} from '../../annotator/selected-media-item-provider.component';
import { SEGMENT_ANYTHING_ENCODING_QUERY_KEY_PREFIX } from '../../annotator/tools/segment-anything-tool/use-segment-anything.hook';
import { useSelectDatasetItem } from '../gallery/hooks/use-select-dataset-item.hook';
import { AnnotatorProviders } from './annotator-providers.component';
import { AnnotatorContainer } from './annotator.component';
import { useAnnotationsQuery } from './api/use-annotations-query';
import { SIDEBAR_WIDTH } from './constants';
import { SidebarItems } from './sidebar-items/sidebar-items.component';
import { useAnnotatorMediaTransition } from './use-annotator-media-transition.hook';
import { getInitialAnnotations, useAnnotatorMode } from './utils';

type MediaPreviewProps = {
    close: () => void;
    onSelectedMediaItem: (item: Media) => void;
};

type MediaPreviewContentProps = {
    items: Media[];
    onClose: () => void;
    onSelectedMediaItem: (item: Media) => void;
    isFetchingNextPage: boolean;
    fetchNextPage: () => void;
    isMediaItemReviewedById: (mediaItemId: string) => boolean;
};

type MediaPreviewPanelsProps = {
    mode: AnnotatorMode;
    changeAnnotatorMode: (mode: AnnotatorMode) => void;
    items: Media[];
    onClose: () => void;
    onSelectedMediaItem: (item: Media) => void;
    isFetchingNextPage: boolean;
    fetchNextPage: () => void;
    isMediaItemReviewedById: (mediaItemId: string) => boolean;
    isCurrentMediaReviewed: boolean;
    subset: DatasetSubset;
};

const MediaPreviewPanels = ({
    mode,
    subset,
    changeAnnotatorMode,
    items,
    onClose,
    onSelectedMediaItem,
    isFetchingNextPage,
    fetchNextPage,
    isMediaItemReviewedById,
    isCurrentMediaReviewed,
}: MediaPreviewPanelsProps) => {
    const { mediaItem } = useSelectedMediaItem();
    const handleMediaTransition = useAnnotatorMediaTransition({ onSelectedMediaItem });

    return (
        <>
            <AnnotatorContainer
                mode={mode}
                items={items}
                subset={subset}
                onClose={onClose}
                isUserReviewed={isCurrentMediaReviewed}
                changeAnnotatorMode={changeAnnotatorMode}
                onSelectedMediaItem={handleMediaTransition}
            />

            <View gridArea={'aside'}>
                <SidebarItems
                    items={items}
                    mediaItem={mediaItem}
                    isFetchingNextPage={isFetchingNextPage}
                    fetchNextPage={fetchNextPage}
                    isUserReviewed={isMediaItemReviewedById}
                    onSelectedMediaItem={handleMediaTransition}
                />
            </View>
        </>
    );
};

/**
 * Aborts inference and embedding requests if the user closes the annotator dialog.
 *
 * It won't stop the server side processing but it will prevent the client from waiting for the results.
 */
const useCancelInferenceOnUnmount = () => {
    const queryClient = useQueryClient();
    const projectId = useProjectIdentifier();

    useEffect(() => {
        return () => {
            void queryClient.cancelQueries({ queryKey: getMediaPredictionsQueryKeyPrefix(projectId) });
            void queryClient.cancelQueries({ queryKey: SEGMENT_ANYTHING_ENCODING_QUERY_KEY_PREFIX });
        };
    }, [queryClient, projectId]);
};

const MediaPreviewContent = ({
    items,
    onSelectedMediaItem,
    onClose,
    isFetchingNextPage,
    fetchNextPage,
    isMediaItemReviewedById,
}: MediaPreviewContentProps) => {
    const { mediaItem } = useSelectedMediaItem();
    const { selectedModel, selectedDevice, confidenceThreshold } = usePredictionSetup();

    useCancelInferenceOnUnmount();

    const { data: annotationsData } = useAnnotationsQuery(mediaItem);
    const { data: predictionsData } = useMediaPredictions({
        mediaId: mediaItem.id,
        selectedModel,
        device: selectedDevice,
        confidenceThreshold,
        range: isVideoFrame(mediaItem)
            ? { start_frame: mediaItem.frame_number, end_frame: mediaItem.frame_number, stride: mediaItem.frame_stride }
            : null,
    });

    const isCurrentMediaReviewed = annotationsData?.user_reviewed ?? false;
    const subset: DatasetSubset = annotationsData?.subset ?? 'unassigned';

    const initialAnnotations = useMemo(() => {
        return getInitialAnnotations(isCurrentMediaReviewed, annotationsData?.annotations ?? []);
    }, [isCurrentMediaReviewed, annotationsData?.annotations]);

    const initialPredictions = useMemo(() => {
        return predictionsData?.flatMap((predictionData) => predictionData.prediction) ?? [];
    }, [predictionsData]);

    const [mode, setMode] = useAnnotatorMode();

    return (
        <ToolProvider>
            <AnnotatorProviders
                mediaItem={mediaItem}
                initialAnnotationsDTO={initialAnnotations}
                initialPredictionsDTO={initialPredictions}
                isUserReviewed={isCurrentMediaReviewed}
                mode={mode}
            >
                <MediaPreviewPanels
                    mode={mode}
                    changeAnnotatorMode={setMode}
                    items={items}
                    onClose={onClose}
                    onSelectedMediaItem={onSelectedMediaItem}
                    isFetchingNextPage={isFetchingNextPage}
                    fetchNextPage={fetchNextPage}
                    isMediaItemReviewedById={isMediaItemReviewedById}
                    isCurrentMediaReviewed={isCurrentMediaReviewed}
                    subset={subset}
                />
            </AnnotatorProviders>
        </ToolProvider>
    );
};

export const MediaPreview = ({ close, onSelectedMediaItem }: MediaPreviewProps) => {
    const { items, isFetchingNextPage, fetchNextPage, isMediaItemReviewedById } = useDatasetMediaWithReviewStatus();

    // Reading the route param unmounts the content on close, without waiting for the dialog's exit animation.
    const { selectedMediaItem } = useSelectDatasetItem();

    return (
        <Dialog
            UNSAFE_style={{
                backgroundColor: 'var(--spectrum-global-color-gray-50)',
                '--spectrum-dialog-padding-x': 'var(--spectrum-global-dimension-size-250)',
                '--spectrum-dialog-padding-y': 'var(--spectrum-global-dimension-size-250)',
            }}
        >
            <Content>
                <Grid
                    gap='size-125'
                    width='100%'
                    height='100%'
                    rows='auto 1fr auto auto'
                    columns={['size-700', 'minmax(0, 1fr)', SIDEBAR_WIDTH]}
                    areas={[
                        'header header aside',
                        'toolbar canvas aside',
                        'toolbar video-toolbar aside',
                        'toolbar bottom aside',
                    ]}
                >
                    {selectedMediaItem !== null && (
                        <SelectedMediaItemProvider mediaItem={selectedMediaItem}>
                            <PredictionsSetupProvider>
                                <MediaPreviewContent
                                    items={items}
                                    onClose={close}
                                    onSelectedMediaItem={onSelectedMediaItem}
                                    isFetchingNextPage={isFetchingNextPage}
                                    fetchNextPage={fetchNextPage}
                                    isMediaItemReviewedById={isMediaItemReviewedById}
                                />
                            </PredictionsSetupProvider>
                        </SelectedMediaItemProvider>
                    )}
                </Grid>
            </Content>
        </Dialog>
    );
};
