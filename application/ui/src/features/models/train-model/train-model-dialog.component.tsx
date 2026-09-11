// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { toast } from '@/components/toast/toast.component';
import { Trans, useTranslation } from '@/i18n';
import { Button, ButtonGroup, Content, Dialog, Divider, Flex, Footer, Heading, InlineAlert, Text } from '@geti-ui/ui';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { Link, useMatch } from 'react-router-dom';

import { paths } from '../../../constants/paths';
import { AdvancedSettings } from './advanced-settings/advanced-settings.component';
import { BasicTrainModelContent } from './basic-train-model-content.component';
import { useTrainModel } from './hooks/use-train-model';
import { useTrainModelDisabledReason } from './hooks/use-train-model-disabled-reason';
import { TrainModelDialogLayout } from './train-model-dialog-layout.component';
import { useTrainModelState } from './train-model-provider.component';

type TrainModelDialogProps = {
    onClose: () => void;
};

export const TrainModelDialog = ({ onClose }: TrainModelDialogProps) => {
    const { t } = useTranslation();
    const {
        selectedTrainingDevice,
        resolvedModelArchitectureId,
        isAdvancedSettingsMode,
        onToggleAdvancedSettingsMode,
        trainingConfiguration,
    } = useTrainModelState();
    const projectId = useProjectIdentifier();
    const isModelsPage = useMatch(paths.project.models.pattern);
    const trainingDisabledReason = useTrainModelDisabledReason().reason;
    const isTrainingDisabled = trainingDisabledReason !== undefined;

    const { trainModel, isPending } = useTrainModel();

    const isStartButtonDisabled =
        isTrainingDisabled || resolvedModelArchitectureId === null || selectedTrainingDevice === null || isPending;

    const isAdvancedSettingsModeDisabled = resolvedModelArchitectureId === null || trainingConfiguration === undefined;

    const handleTrainModel = () => {
        trainModel({
            onSuccess: () => {
                onClose();

                toast({
                    message: isModelsPage ? (
                        <Text>{t('models.training.setup.toast.trainingStarted')}</Text>
                    ) : (
                        <Flex alignItems={'center'} gap={'size-50'} wrap={'wrap'}>
                            <Text>
                                <Trans
                                    i18nKey='models.training.setup.toast.trainingStartedWithLink'
                                    components={{
                                        link: <Link to={paths.project.models({ projectId })} viewTransition />,
                                    }}
                                />
                            </Text>
                        </Flex>
                    ),
                    type: 'success',
                });
            },
        });
    };

    return (
        <Dialog width={'clamp(800px, 50vw, 1150px)'} height={isAdvancedSettingsMode ? '80vh' : undefined}>
            <Heading>{t('models.training.setup.dialog.title')}</Heading>

            <Divider size={'S'} />

            <Content>
                <TrainModelDialogLayout>
                    {isAdvancedSettingsMode ? <AdvancedSettings /> : <BasicTrainModelContent />}
                </TrainModelDialogLayout>
            </Content>

            <Divider size={'S'} />

            <Footer>
                <Flex alignItems={'center'} marginBottom={'size-200'}>
                    {isTrainingDisabled ? (
                        <InlineAlert variant={'notice'}>
                            <Heading>{t('models.training.setup.dialog.disabledReasonTitle')}</Heading>
                            <Content>{trainingDisabledReason}</Content>
                        </InlineAlert>
                    ) : null}
                </Flex>

                <ButtonGroup marginStart={'auto'}>
                    <Button variant={'secondary'} onPress={onClose}>
                        {t('models.training.setup.dialog.cancel')}
                    </Button>
                    {isAdvancedSettingsMode ? (
                        <Button
                            variant={'primary'}
                            onPress={() => onToggleAdvancedSettingsMode(!isAdvancedSettingsMode)}
                        >
                            {t('models.training.setup.dialog.back')}
                        </Button>
                    ) : (
                        <Button
                            variant={'primary'}
                            isDisabled={isAdvancedSettingsModeDisabled}
                            onPress={() => onToggleAdvancedSettingsMode(!isAdvancedSettingsMode)}
                        >
                            {t('models.training.setup.dialog.advancedSettings')}
                        </Button>
                    )}

                    <Button
                        variant={'accent'}
                        onPress={handleTrainModel}
                        isDisabled={isStartButtonDisabled}
                        isPending={isPending}
                    >
                        {t('models.training.setup.dialog.start')}
                    </Button>
                </ButtonGroup>
            </Footer>
        </Dialog>
    );
};
