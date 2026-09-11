// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { ActionButton, Flex, PressableElement, Tooltip, TooltipTrigger } from '@geti-ui/ui';
import { Pause, Play, SoundOff, SoundOn, StepBackward, StepForward } from '@geti-ui/ui/icons';

import { AnnotatorMode } from '../../../../shared/annotator/annotator-mode';
import { useIsFetchingPredictions } from '../../api/use-media-predictions';
import { getVideoErrorMessage } from '../video-player-error';
import { useVideoPlayer } from '../video-player-provider.component';

type VideoControlsProps = {
    mode: AnnotatorMode;
};

export const VideoControls = ({ mode }: VideoControlsProps) => {
    const { t } = useTranslation();
    const { isMuted, toggleMute, videoControls, videoFrame, videoError } = useVideoPlayer();
    const { isPlaying, play, pause, previousFrame, nextFrame, canSelectPreviousFrame, canSelectNextFrame } =
        videoControls;

    const isLoadingPredictions = useIsFetchingPredictions(videoFrame.id) && mode === 'prediction';

    const isPlayDisabled = videoError !== null || isLoadingPredictions;

    return (
        <Flex alignItems={'center'} gap={'size-100'}>
            <Flex alignItems={'center'}>
                <ActionButton
                    isQuiet
                    aria-label={'Go to previous frame'}
                    isDisabled={!canSelectPreviousFrame}
                    onPress={previousFrame}
                >
                    <StepBackward />
                </ActionButton>
                {isPlaying ? (
                    <ActionButton isQuiet aria-label={'Pause video'} onPress={pause}>
                        <Pause />
                    </ActionButton>
                ) : (
                    <TooltipTrigger delay={500} isDisabled={videoError === null}>
                        <PressableElement>
                            <ActionButton isQuiet aria-label={'Play video'} onPress={play} isDisabled={isPlayDisabled}>
                                <Play />
                            </ActionButton>
                        </PressableElement>
                        <Tooltip>{getVideoErrorMessage(videoError, t)}</Tooltip>
                    </TooltipTrigger>
                )}
                <ActionButton
                    isQuiet
                    aria-label={'Go to next frame'}
                    isDisabled={!canSelectNextFrame}
                    onPress={nextFrame}
                >
                    <StepForward />
                </ActionButton>
            </Flex>
            <ActionButton isQuiet aria-label={isMuted ? 'Unmute audio' : 'Mute audio'} onPress={toggleMute}>
                {isMuted ? <SoundOff /> : <SoundOn />}
            </ActionButton>
        </Flex>
    );
};
