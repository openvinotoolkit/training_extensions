// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { useTranslation } from '@/i18n';
import { ActionButton, Divider, Flex, Text, View } from '@geti-ui/ui';
import { ChevronDownLight } from '@geti-ui/ui/icons';
import { clsx } from 'clsx';

import type { AnnotatorMode } from '../../../../shared/annotator/annotator-mode';
import { Toolbar } from '../../../dataset/media-preview/toolbar-container/toolbar-container.component';
import { PREDICTION_CHUNK_SIZE, usePrefetchVideoFramesPredictions } from '../api/use-video-frames-predictions';
import { useVideoPlayer } from '../video-player-provider.component';
import { FrameStep } from './frame-step/frame-step.component';
import { PlaybackSpeedSlider } from './playback-rate.component';
import { VideoAnnotator } from './video-annotator/video-annotator.component';
import { VideoPlayerSlider } from './video-annotator/video-timeline/video-player-slider/video-player-slider.component';
import { VideoControls } from './video-controls.component';
import { VideoDuration } from './video-duration.component';

import classes from './video-toolbar.module.scss';

type VideoToolbarProps = {
    mode: AnnotatorMode;
};

export const VideoToolbar = ({ mode }: VideoToolbarProps) => {
    const { t } = useTranslation();
    const { videoFrame, step, changeStep, videoControls } = useVideoPlayer();
    const [isExpanded, setIsExpanded] = useState(false);

    // Prefetch predictions for the video segments that are displayed under video timeline
    usePrefetchVideoFramesPredictions({
        frameNumber: videoFrame.frame_number,
        frameSkip: step,
        chunkSize: PREDICTION_CHUNK_SIZE,
    });

    return (
        <Toolbar.Container>
            <Toolbar.Section>
                <View paddingX={'size-100'}>
                    <Flex alignItems={'center'} justifyContent={'space-between'} gap={'size-200'}>
                        <Flex alignItems={'center'} gap={'size-200'}>
                            {isExpanded && <Text>{t('annotator.video.frames.label')}</Text>}

                            <VideoControls mode={mode} />
                            <VideoDuration videoFrame={videoFrame} />

                            {isExpanded && (
                                <Flex alignItems={'center'} gap={'size-100'}>
                                    <Divider orientation={'vertical'} size={'S'} />

                                    <FrameStep
                                        step={step}
                                        onChangeStep={changeStep}
                                        isDisabled={videoControls.isPlaying}
                                        defaultFps={videoFrame.frame_stride}
                                    />

                                    <PlaybackSpeedSlider />

                                    <Divider orientation={'vertical'} size={'S'} />
                                </Flex>
                            )}
                        </Flex>

                        <Flex alignItems={'center'} gap={'size-100'} flex={isExpanded ? undefined : 1}>
                            {isExpanded ? (
                                <Text>
                                    {t('annotator.video.frames.currentTotal', {
                                        currentFrame: videoFrame.frame_number,
                                        totalFrames: videoFrame.frame_count - 1,
                                    })}
                                </Text>
                            ) : (
                                <View flex={1}>
                                    <VideoPlayerSlider
                                        videoFrame={videoFrame}
                                        step={step}
                                        frameNumber={videoFrame.frame_number}
                                        selectFrame={videoControls.goto}
                                    />
                                </View>
                            )}

                            <ActionButton
                                isQuiet
                                onPress={() => setIsExpanded((prev) => !prev)}
                                aria-label={`${isExpanded ? 'Collapse' : 'Expand'} toolbar`}
                            >
                                <ChevronDownLight
                                    className={clsx(classes.chevronButton, {
                                        [classes.chevronButtonCollapsed]: !isExpanded,
                                    })}
                                />
                            </ActionButton>
                        </Flex>
                    </Flex>
                    {isExpanded && <VideoAnnotator mode={mode} />}
                </View>
            </Toolbar.Section>
        </Toolbar.Container>
    );
};
