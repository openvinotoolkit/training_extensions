// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { toast } from '@/components/toast/toast.component';
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
                        <Text>Model training started successfully.</Text>
                    ) : (
                        <Flex alignItems={'center'} gap={'size-50'} wrap={'wrap'}>
                            <Text>
                                Model training started successfully.{' '}
                                <Link to={paths.project.models({ projectId })} viewTransition>
                                    Open models screen to see progress.
                                </Link>
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
            <Heading>Select a model to train</Heading>

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
                            <Heading>Why can I not start training?</Heading>
                            <Content>{trainingDisabledReason}</Content>
                        </InlineAlert>
                    ) : null}
                </Flex>

                <ButtonGroup marginStart={'auto'}>
                    <Button variant={'secondary'} onPress={onClose}>
                        Cancel
                    </Button>
                    {isAdvancedSettingsMode ? (
                        <Button
                            variant={'primary'}
                            onPress={() => onToggleAdvancedSettingsMode(!isAdvancedSettingsMode)}
                        >
                            Back
                        </Button>
                    ) : (
                        <Button
                            variant={'primary'}
                            isDisabled={isAdvancedSettingsModeDisabled}
                            onPress={() => onToggleAdvancedSettingsMode(!isAdvancedSettingsMode)}
                        >
                            Advanced settings
                        </Button>
                    )}

                    <Button
                        variant={'accent'}
                        onPress={handleTrainModel}
                        isDisabled={isStartButtonDisabled}
                        isPending={isPending}
                    >
                        Start
                    </Button>
                </ButtonGroup>
            </Footer>
        </Dialog>
    );
};
