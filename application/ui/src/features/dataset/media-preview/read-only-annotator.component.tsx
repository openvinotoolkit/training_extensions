// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { DatasetSubset, Media } from '@/api/types';
import { useTranslation } from '@/i18n';
import { ActionButton, Flex, Icon, Text, View } from '@geti-ui/ui';
import { ChevronLeft, ChevronRight, CloseSemiBold } from '@geti-ui/ui/icons';
import { isEmpty } from 'lodash-es';

import { useAnnotationActions } from '../../../shared/annotator/annotation-actions-provider.component';
import type { AnnotatorMode } from '../../../shared/annotator/annotator-mode';
import { ReadOnlyAnnotatorCanvas } from '../../annotator/annotator-canvas/read-only-annotator-canvas';
import { BottomToolbar } from './bottom-toolbar/bottom-toolbar.component';
import { AnnotatorCanvasSettings } from './primary-toolbar/settings/annotator-canvas-settings.component';
import { AnnotatorModes } from './secondary-toolbar/annotator-modes/annotator-modes-toggle.component';
import { Toolbar } from './toolbar-container/toolbar-container.component';

import classes from './read-only-annotator.module.scss';

type ReadOnlyAnnotatorProps = {
    mediaItem: Media;
    image: ImageData;
    onClose: () => void;
    subset: DatasetSubset;
    hasAnnotationStatus?: boolean;
    mode: AnnotatorMode;
    onModeChange?: (mode: AnnotatorMode) => void;
    // Blocks navigation and dims the canvas while the content of the current media item is loading
    isLoading?: boolean;
    // Left undefined when there is no adjacent item to navigate to
    onSelectPreviousMediaItem?: () => void;
    onSelectNextMediaItem?: () => void;
};

type ModesToggleProps = {
    mediaItem: Media;
    mode: AnnotatorMode;
    onModeChange: (mode: AnnotatorMode) => void;
};

const ModesToggle = ({ mediaItem, mode, onModeChange }: ModesToggleProps) => {
    const { initialAnnotations, initialPredictions } = useAnnotationActions();

    return (
        <Toolbar.Container>
            <Toolbar.Section>
                <AnnotatorModes
                    // We want to reset the prediction cue when the media item changes
                    key={mediaItem.id}
                    mode={mode}
                    onModeChange={onModeChange}
                    hasAnnotations={!isEmpty(initialAnnotations)}
                    hasPredictions={!isEmpty(initialPredictions)}
                />
            </Toolbar.Section>
        </Toolbar.Container>
    );
};

type NavigationButtonsProps = {
    onSelectPreviousMediaItem?: () => void;
    onSelectNextMediaItem?: () => void;
    isLoading?: boolean;
};
const NavigationButtons = ({ onSelectPreviousMediaItem, onSelectNextMediaItem, isLoading }: NavigationButtonsProps) => {
    return (
        <>
            <ActionButton
                isQuiet
                aria-label={'Previous media item'}
                isDisabled={onSelectPreviousMediaItem === undefined || isLoading}
                onPress={onSelectPreviousMediaItem}
            >
                <Icon height={'size-150'} width={'size-150'}>
                    <ChevronLeft />
                </Icon>
            </ActionButton>
            <ActionButton
                isQuiet
                aria-label={'Next media item'}
                isDisabled={onSelectNextMediaItem === undefined || isLoading}
                onPress={onSelectNextMediaItem}
            >
                <Icon height={'size-150'} width={'size-150'}>
                    <ChevronRight />
                </Icon>
            </ActionButton>
        </>
    );
};

/**
 * Simplified read-only annotator for viewing annotations.
 *
 * Features:
 * - Read-only canvas (no annotation editing)
 * - Bottom toolbar without hotkeys
 * - No primary toolbar
 * - Optional annotation/prediction toggle and media navigation
 *
 * Note: This component renders into the parent grid layout from MediaPreview.
 * It uses the same gridArea structure as the normal annotator but with fewer elements.
 */
export const ReadOnlyAnnotator = ({
    image,
    mediaItem,
    subset,
    hasAnnotationStatus = true,
    mode = 'annotation',
    onModeChange,
    isLoading = false,
    onSelectPreviousMediaItem,
    onSelectNextMediaItem,
    onClose,
}: ReadOnlyAnnotatorProps) => {
    const { t } = useTranslation();
    const hasMediaNavigation = onSelectPreviousMediaItem !== undefined || onSelectNextMediaItem !== undefined;

    return (
        <>
            <View gridArea={'header'} UNSAFE_className={classes.toolbarContainer}>
                <Flex alignItems={'center'} justifyContent={'space-between'} width={'100%'} gap={'size-100'}>
                    {onModeChange !== undefined && (
                        <ModesToggle mediaItem={mediaItem} mode={mode} onModeChange={onModeChange} />
                    )}

                    <Toolbar.Container marginStart={'auto'}>
                        <Toolbar.Section>
                            <Flex alignItems={'center'}>
                                {hasMediaNavigation && (
                                    <NavigationButtons
                                        onSelectPreviousMediaItem={onSelectPreviousMediaItem}
                                        onSelectNextMediaItem={onSelectNextMediaItem}
                                        isLoading={isLoading}
                                    />
                                )}

                                <ActionButton isQuiet onPress={onClose}>
                                    <Icon height={'size-150'} width={'size-150'}>
                                        <CloseSemiBold />
                                    </Icon>
                                    <Text>{t('annotator.actions.close')}</Text>
                                </ActionButton>
                            </Flex>
                        </Toolbar.Section>
                    </Toolbar.Container>
                </Flex>
            </View>

            <View gridArea={'canvas'} overflow={'hidden'}>
                <AnnotatorCanvasSettings>
                    <ReadOnlyAnnotatorCanvas mediaItem={mediaItem} image={image} isLoadingOverlay={isLoading} />
                </AnnotatorCanvasSettings>
            </View>

            <View gridArea={'bottom'}>
                <BottomToolbar
                    mediaItem={mediaItem}
                    hideHotkeys
                    subset={subset}
                    isReadOnlySubset
                    hasAnnotationStatus={hasAnnotationStatus}
                />
            </View>
        </>
    );
};
