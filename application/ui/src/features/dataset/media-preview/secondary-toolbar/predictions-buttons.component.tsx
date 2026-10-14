// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { ActionButton, Icon, Text } from '@geti-ui/ui';
import { Checkmark, Edit } from '@geti-ui/ui/icons';

import {
    useAnnotationCommands,
    useAnnotations,
} from '../../../../modules/annotator/annotation-document-provider.component';
import { convertPredictionToAnnotation } from '../../../../modules/annotator/annotations/utils';
import type { AnnotatorMode } from '../../../../modules/annotator/annotator-mode';

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
            <Text>{t('common.actions.edit')}</Text>
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
    const { replaceAnnotations } = useAnnotationCommands();
    const { annotations } = useAnnotations();

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
