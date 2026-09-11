// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useMemo } from 'react';

import { useTranslation } from '@/i18n';
import { ActionButton, DialogTrigger, Flex, Slider, Text, Tooltip, TooltipTrigger, View } from '@geti-ui/ui';

import { ReactComponent as PlayRate } from '../../../../assets/icons/play-rate.svg';
import { useVideoPlayer } from '../video-player-provider.component';

type PlaybackRate = {
    value: number;
    label: string;
    key: number;
};

const MIN_RATE = 1;
const MAX_RATE = 3;

export const PlaybackSpeedSlider = () => {
    const { t } = useTranslation();
    const { playbackRate, changePlaybackRate } = useVideoPlayer();

    const playbackSpeedLabel = t('annotator.video.playback.speed');

    const availablePlaybackRatesMapping = useMemo((): Record<number, PlaybackRate> => {
        return {
            [1]: {
                value: 0.25,
                label: t('annotator.video.playback.slower'),
                key: 1,
            },
            [2]: {
                value: 0.5,
                label: t('annotator.video.playback.slow'),
                key: 2,
            },
            [3]: {
                value: 1,
                label: t('annotator.video.playback.normal'),
                key: 3,
            },
        };
    }, [t]);

    const selectedPlaybackRate =
        Object.values(availablePlaybackRatesMapping).find(({ value }) => value === playbackRate)?.key ?? MAX_RATE;

    return (
        <DialogTrigger type='popover'>
            <TooltipTrigger placement={'bottom'}>
                <ActionButton isQuiet aria-label={'Change playback speed'}>
                    <View padding={'size-100'} width={'size-1000'}>
                        <Flex alignItems={'center'} gap={'size-100'}>
                            <PlayRate />
                            <Text>{availablePlaybackRatesMapping[selectedPlaybackRate].value}x</Text>
                        </Flex>
                    </View>
                </ActionButton>
                <Tooltip>{playbackSpeedLabel}</Tooltip>
            </TooltipTrigger>

            <View padding={'size-200'}>
                <Slider
                    id={'playback-speed'}
                    minValue={MIN_RATE}
                    maxValue={MAX_RATE}
                    step={1}
                    label={playbackSpeedLabel}
                    getValueLabel={(value) => availablePlaybackRatesMapping[value].label}
                    aria-label={'Playback'}
                    value={selectedPlaybackRate}
                    onChange={(value) => {
                        changePlaybackRate(availablePlaybackRatesMapping[value].value);
                    }}
                    isFilled
                />
            </View>
        </DialogTrigger>
    );
};
