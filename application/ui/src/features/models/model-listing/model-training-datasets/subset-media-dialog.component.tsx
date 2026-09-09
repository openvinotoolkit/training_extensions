// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useMemo, useState } from 'react';

import type { DatasetRevisionItem } from '@/api/types';
import { Content, Dialog, Grid } from '@geti-ui/ui';
import { usePipeline } from 'hooks/api/pipeline.hook';

import type { AnnotatorMode } from '../../../../shared/annotator/annotator-mode';
import { useIsFetchingMediaPredictions, useMediaPredictions } from '../../../annotator/api/use-media-predictions';
import {
    SelectedMediaItemProvider,
    useSelectedMediaItem,
} from '../../../annotator/selected-media-item-provider.component';
import { useAnnotationsQuery } from '../../../dataset/media-preview/api/use-annotations-query';
import { ReadOnlyAnnotatorProviders } from '../../../dataset/media-preview/read-only-annotator-providers.component';
import { ReadOnlyAnnotator } from '../../../dataset/media-preview/read-only-annotator.component';
import { getInitialAnnotations } from '../../../dataset/media-preview/utils';
import { type SelectableModel } from '../../utils';
import { datasetRevisionItemToMedia } from './utils';

type SubsetMediaDialogProps = {
    item: DatasetRevisionItem;
    onClose: () => void;
    selectedModel: SelectableModel | undefined;
    onSelectPreviousMediaItem?: () => void;
    onSelectNextMediaItem?: () => void;
};

type SubsetMediaDialogContentProps = SubsetMediaDialogProps & {
    mode: AnnotatorMode;
    onModeChange: (mode: AnnotatorMode) => void;
};

const SubsetMediaDialogContent = ({
    item,
    onClose,
    selectedModel,
    mode,
    onModeChange,
    onSelectPreviousMediaItem,
    onSelectNextMediaItem,
}: SubsetMediaDialogContentProps) => {
    const { mediaItem, image } = useSelectedMediaItem();
    const { data: annotationsData, isFetching: isFetchingAnnotations } = useAnnotationsQuery(mediaItem);
    const { data: pipeline } = usePipeline();

    const isPredictionMode = mode === 'prediction';

    const { data: predictionsData } = useMediaPredictions({
        mediaId: mediaItem.id,
        selectedModel,
        device: pipeline.device,
        confidenceThreshold: selectedModel?.optimalConfidenceThreshold ?? null,
        range: null,
    });

    const isFetchingPredictions = useIsFetchingMediaPredictions({
        mediaId: mediaItem.id,
        selectedModel,
        device: pipeline.device,
        confidenceThreshold: selectedModel?.optimalConfidenceThreshold ?? null,
        range: null,
    });

    // Inference started in prediction mode keeps running after switching back, it must not block the annotations
    const isLoading = isFetchingAnnotations || (isPredictionMode && isFetchingPredictions);

    const annotationsDTO = annotationsData?.annotations ?? [];
    const isUserReviewed = annotationsData?.user_reviewed ?? false;

    const initialPredictionsDTO = useMemo(() => {
        return predictionsData?.flatMap((predictionData) => predictionData.prediction) ?? [];
    }, [predictionsData]);

    return (
        <ReadOnlyAnnotatorProviders
            key={mediaItem.id}
            mediaItem={mediaItem}
            initialAnnotationsDTO={getInitialAnnotations(isUserReviewed, annotationsDTO)}
            initialPredictionsDTO={initialPredictionsDTO}
            isUserReviewed={isUserReviewed}
            mode={mode}
        >
            <ReadOnlyAnnotator
                image={image}
                mediaItem={mediaItem}
                onClose={onClose}
                subset={item.subset}
                hasAnnotationStatus={false}
                mode={mode}
                onModeChange={selectedModel === undefined ? undefined : onModeChange}
                isLoading={isLoading}
                onSelectPreviousMediaItem={onSelectPreviousMediaItem}
                onSelectNextMediaItem={onSelectNextMediaItem}
            />
        </ReadOnlyAnnotatorProviders>
    );
};

export const SubsetMediaDialog = ({
    item,
    onClose,
    selectedModel,
    onSelectPreviousMediaItem,
    onSelectNextMediaItem,
}: SubsetMediaDialogProps) => {
    const mediaItem = datasetRevisionItemToMedia(item);
    const [mode, setMode] = useState<AnnotatorMode>('annotation');

    return (
        <Dialog>
            <Content>
                <Grid
                    gap='size-125'
                    width='100%'
                    height='100%'
                    rows='auto 1fr auto'
                    columns={['1fr']}
                    areas={['header', 'canvas', 'bottom']}
                >
                    <SelectedMediaItemProvider mediaItem={mediaItem}>
                        <SubsetMediaDialogContent
                            item={item}
                            onClose={onClose}
                            selectedModel={selectedModel}
                            mode={mode}
                            onModeChange={setMode}
                            onSelectPreviousMediaItem={onSelectPreviousMediaItem}
                            onSelectNextMediaItem={onSelectNextMediaItem}
                        />
                    </SelectedMediaItemProvider>
                </Grid>
            </Content>
        </Dialog>
    );
};
