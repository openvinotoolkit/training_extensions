// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { BoundingBox, Polygon, SegmentAnythingIcon, Selector } from '@geti-ui/ui/icons';

import { ReactComponent as MagneticLasso } from '../../../../assets/icons/magnetic-lasso.svg';
import BoundingBoxImg from '../../../../assets/tools/bounding-box.webp';
import MagneticLassoImg from '../../../../assets/tools/magnetic-lasso.webp';
import PolygonImg from '../../../../assets/tools/polygon.webp';
import SAMDetectionImg from '../../../../assets/tools/sam-detection.webp';
import SAMSegmentationImg from '../../../../assets/tools/sam-segmentation.webp';
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

    const taskToolConfig: Record<string, ToolConfig[]> = {
        classification: [],
        detection: [selectionToolConfig, boundingBoxToolConfig, autoSegmentationDetectionConfig],
        instance_segmentation: [
            selectionToolConfig,
            polygonToolConfig,
            magneticLassoToolConfig,
            autoSegmentationConfig,
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
