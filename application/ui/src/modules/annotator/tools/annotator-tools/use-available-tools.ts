// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { MagneticLasso } from '@/assets/icons';
import { BoundingBoxImg, MagneticLassoImg, PolygonImg, SAMDetectionImg, SAMSegmentationImg } from '@/assets/tools';
import { useTranslation } from '@/i18n';
import { BoundingBox, DetectionTool, Polygon, SegmentAnythingIcon, Selector } from '@geti-ui/ui/icons';

import { FEATURE_FLAGS } from '../../../../constants/feature-flags';
import { useProjectTask } from '../../../../hooks/use-project-task.hook';
import { HOTKEYS } from '../../../../shared/hotkeys-definition';
import { useSelectedMediaItem } from '../../selected-media-item-provider.component';
import { ToolConfig } from '../interface';
import { canRasteriseAtFullSize } from '../utils';

export const useAvailableTools = (): ToolConfig[] => {
    const { t } = useTranslation();
    const taskType = useProjectTask();
    const { mediaItem } = useSelectedMediaItem();

    const selectionToolConfig: ToolConfig = {
        type: 'selection',
        icon: Selector,
        hotkey: HOTKEYS.selectionTool,
        label: t('annotator.tools.selection.label'),
        ariaLabel: 'Selection',
    };

    const boundingBoxToolConfig: ToolConfig = {
        type: 'bounding-box',
        icon: BoundingBox,
        hotkey: HOTKEYS.boundingBoxTool,
        label: t('annotator.tools.boundingBox.label'),
        ariaLabel: 'Bounding box',
        tooltip: {
            img: BoundingBoxImg,
            description: t('annotator.tools.boundingBox.description'),
        },
    };

    const autoSegmentationDetectionConfig: ToolConfig = {
        type: 'sam',
        icon: SegmentAnythingIcon,
        hotkey: HOTKEYS.autoSegmentation,
        label: t('annotator.tools.autoSegmentation.label'),
        ariaLabel: 'Auto segmentation',
        tooltip: {
            img: SAMDetectionImg,
            description: t('annotator.tools.autoSegmentation.detectionDescription'),
        },
    };

    const autoSegmentationConfig: ToolConfig = {
        type: 'sam',
        icon: SegmentAnythingIcon,
        hotkey: HOTKEYS.autoSegmentation,
        label: t('annotator.tools.autoSegmentation.label'),
        ariaLabel: 'Auto segmentation',
        tooltip: {
            img: SAMSegmentationImg,
            description: t('annotator.tools.autoSegmentation.segmentationDescription'),
        },
    };

    const polygonToolConfig: ToolConfig = {
        type: 'polygon',
        icon: Polygon,
        hotkey: HOTKEYS.polygonTool,
        label: t('annotator.tools.polygon.label'),
        ariaLabel: 'Polygon',
        tooltip: {
            img: PolygonImg,
            description: t('annotator.tools.polygon.description'),
        },
    };

    const magneticLassoToolConfig: ToolConfig = {
        type: 'magnetic-lasso',
        icon: MagneticLasso,
        hotkey: HOTKEYS.magneticLassoTool,
        label: t('annotator.tools.magneticLasso.label'),
        ariaLabel: 'Magnetic Lasso',
        tooltip: {
            img: MagneticLassoImg,
            description: t('annotator.tools.magneticLasso.description'),
        },
    };

    const ssimToolConfig: ToolConfig = {
        type: 'ssim',
        icon: DetectionTool,
        hotkey: HOTKEYS.ssimTool,
        label: t('annotator.tools.ssim.label'),
        ariaLabel: 'Detection assistant',
        tooltip: {
            description: t('annotator.tools.ssim.description'),
        },
    };

    const ssimTools = FEATURE_FLAGS.SSIM_TOOL ? [ssimToolConfig] : [];

    const taskToolConfig: Record<string, ToolConfig[]> = {
        classification: [],
        detection: [selectionToolConfig, boundingBoxToolConfig, autoSegmentationDetectionConfig, ...ssimTools],
        instance_segmentation: [
            selectionToolConfig,
            polygonToolConfig,
            magneticLassoToolConfig,
            autoSegmentationConfig,
            ...ssimTools,
        ],
    };

    // Disable smart tools (SAM, magnetic lasso, SSIM) for oversized media.
    if (!canRasteriseAtFullSize(mediaItem.width, mediaItem.height)) {
        return taskToolConfig[taskType].filter(
            (tool) => tool.type !== 'sam' && tool.type !== 'magnetic-lasso' && tool.type !== 'ssim'
        );
    }

    return taskToolConfig[taskType];
};
