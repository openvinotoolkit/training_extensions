// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode, useState } from 'react';

import { useTranslation } from '@/i18n';
import { Flex, StatusLight, View } from '@geti-ui/ui';

import { type AnnotatorMode } from '../../../../../shared/annotator/annotator-mode';

import classes from './annotator-modes.module.scss';

interface ToggleButtonProps {
    children: ReactNode;
    isActive: boolean;
    onClick: () => void;
}

const ToggleButton = ({ children, isActive, onClick }: ToggleButtonProps) => {
    return (
        <button aria-pressed={isActive} onClick={onClick} className={classes.toggleButton}>
            {children}
        </button>
    );
};

interface ToggleButtonWithCueProps extends ToggleButtonProps {
    showCue: boolean;
    cueLabel: string;
}

const ToggleButtonWithCue = ({ showCue, cueLabel, onClick, isActive, children }: ToggleButtonWithCueProps) => {
    return (
        <Flex alignItems={'center'} position={'relative'}>
            <ToggleButton isActive={isActive} onClick={onClick}>
                {children}
            </ToggleButton>
            {showCue && (
                <StatusLight
                    variant={'info'}
                    role={'status'}
                    aria-label={cueLabel}
                    UNSAFE_className={classes.availabilityCue}
                />
            )}
        </Flex>
    );
};

type AnnotatorModesProps = {
    mode: AnnotatorMode;
    onModeChange: (mode: AnnotatorMode) => void;
    hasAnnotations: boolean;
    hasPredictions: boolean;
};

export const AnnotatorModes = ({ mode, onModeChange, hasAnnotations, hasPredictions }: AnnotatorModesProps) => {
    const { t } = useTranslation();
    const [dismissedCues, setDismissedCues] = useState<Set<Extract<AnnotatorMode, 'prediction'>>>(new Set());

    const shouldDisplayPredictionCue =
        mode === 'annotation' && !hasAnnotations && hasPredictions && !dismissedCues.has('prediction');

    const handleModeChange = (nextMode: AnnotatorMode) => {
        if (mode === nextMode) {
            return;
        }

        setDismissedCues((prevDismissedCues) => {
            const nextDismissedCues = new Set(prevDismissedCues);

            // user is leaving the current mode — if it had content they've seen it
            const currentHasContent = mode === 'prediction' && hasPredictions;
            if (currentHasContent) {
                nextDismissedCues.add(mode);
            }

            // user is switching to the next mode — if it has content they're about to see it
            const nextHasContent = nextMode === 'prediction' && hasPredictions;
            if (nextHasContent) {
                nextDismissedCues.add(nextMode);
            }

            return nextDismissedCues;
        });

        onModeChange(nextMode);
    };

    return (
        <View backgroundColor={'gray-200'} padding={'size-50'} borderRadius={'regular'}>
            <Flex
                width={'100%'}
                height={'100%'}
                gap={'size-100'}
                alignItems={'center'}
                data-testid={'annotator-modes-id'}
            >
                <ToggleButton isActive={mode === 'annotation'} onClick={() => handleModeChange('annotation')}>
                    {t('annotator.modes.annotation')}
                </ToggleButton>
                <ToggleButtonWithCue
                    isActive={mode === 'prediction'}
                    onClick={() => handleModeChange('prediction')}
                    showCue={shouldDisplayPredictionCue}
                    cueLabel={'Prediction available'}
                >
                    {t('annotator.modes.prediction')}
                </ToggleButtonWithCue>
            </Flex>
        </View>
    );
};
