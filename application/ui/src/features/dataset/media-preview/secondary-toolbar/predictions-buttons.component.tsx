// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { ActionButton, Icon, Text } from '@geti-ui/ui';
import { Checkmark, Edit } from '@geti-ui/ui/icons';

import { useAnnotationActions } from '../../../../shared/annotator/annotation-actions-provider.component';
import type { AnnotatorMode } from '../../../../shared/annotator/annotator-mode';
import { convertPredictionToAnnotation } from '../../../annotator/annotations/utils';

type EditPredictionButtonProps = {
    isDisabled: boolean;
    onEditPrediction: () => void;
};

const EditPredictionButton = ({ isDisabled, onEditPrediction }: EditPredictionButtonProps) => {
    const { t } = useTranslation();

    return (
        <ActionButton isQuiet onPress={onEditPrediction} isDisabled={isDisabled} aria-label={'Edit prediction'}>
            <Icon>
                <Edit />
            </Icon>
            <Text>{t('annotator.predictions.edit')}</Text>
        </ActionButton>
    );
};

type PredictionButtonsProps = {
    onSubmit: () => void;
    isDisabled: boolean;
    onModeChange: (mode: AnnotatorMode) => void;
};

export const PredictionButtons = ({ onSubmit, onModeChange, isDisabled }: PredictionButtonsProps) => {
    const { t } = useTranslation();
    const { replaceAnnotations, annotations } = useAnnotationActions();

    const handleEditPrediction = () => {
        onModeChange('annotation');
        replaceAnnotations(annotations.map(convertPredictionToAnnotation));
    };

    return (
        <>
            <ActionButton isQuiet onPress={onSubmit} isDisabled={isDisabled}>
                <Checkmark />
                <Text>{t('annotator.predictions.confirm')}</Text>
            </ActionButton>

            <EditPredictionButton onEditPrediction={handleEditPrediction} isDisabled={isDisabled} />
        </>
    );
};
