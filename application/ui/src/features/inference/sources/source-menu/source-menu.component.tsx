// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { $api } from '@/api';
import { toast } from '@/components/toast/toast.component';
import { useTranslation } from '@/i18n';
import {
    ActionButton,
    Button,
    ButtonGroup,
    Content,
    Dialog,
    DialogContainer,
    Divider,
    Heading,
    Item,
    Key,
    Menu,
    MenuTrigger,
    Text,
} from '@geti-ui/ui';
import { MoreMenu } from '@geti-ui/ui/icons';
import { useDisablePipeline } from 'hooks/api/pipeline.hook';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { useWebRTCConnection } from '../../stream/web-rtc-connection-provider';

type DisconnectSourceWarningDialogProps = {
    onCancel: () => void;
    onDisconnect: () => void;
    name: string;
    isPending: boolean;
};

const DisconnectSourceWarningDialog = ({
    name,
    onCancel,
    isPending,
    onDisconnect,
}: DisconnectSourceWarningDialogProps) => {
    const { t } = useTranslation();

    return (
        <Dialog>
            <Heading>{t('inference.sources.menu.disconnectDialog.title', { name })}</Heading>
            <Divider />
            <Content>
                <Text>{t('inference.sources.menu.disconnectDialog.message')}</Text>
            </Content>
            <ButtonGroup>
                <Button variant={'secondary'} onPress={onCancel}>
                    {t('inference.sources.menu.disconnectDialog.cancel')}
                </Button>
                <Button variant={'accent'} onPress={onDisconnect} isDisabled={isPending}>
                    {t('inference.sources.menu.disconnectDialog.confirm')}
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};

const SOURCE_MENU_OPTIONS = {
    CONNECT: 'connect',
    DISCONNECT: 'disconnect',
    REMOVE: 'remove',
    EDIT: 'edit',
    TEST: 'test',
};

export type SourceMenuProps = {
    id: string;
    name: string;
    isConnected: boolean;
    onEdit: () => void;
    isPipelineRunning: boolean;
    onTest: () => Promise<void>;
};

export const SourceMenu = ({ id, name, isConnected, onEdit, isPipelineRunning, onTest }: SourceMenuProps) => {
    const { t } = useTranslation();
    const project_id = useProjectIdentifier();
    const [isDisconnectConfirmationDialogVisible, setIsDisconnectConfirmationDialogVisible] = useState<boolean>(false);
    const disablePipelineMutation = useDisablePipeline();
    const { stop } = useWebRTCConnection();

    const updatePipeline = $api.useMutation('patch', '/api/projects/{project_id}/pipeline', {
        meta: {
            invalidateQueries: [
                ['get', '/api/sources'],
                ['get', '/api/projects/{project_id}/pipeline', { params: { path: { project_id } } }],
            ],
        },
    });

    const removeSource = $api.useMutation('delete', '/api/sources/{source_id}', {
        meta: {
            invalidateQueries: [['get', '/api/sources']],
        },
    });

    const handleOnAction = (option: Key) => {
        switch (option) {
            case SOURCE_MENU_OPTIONS.CONNECT:
                handleConnect();
                break;
            case SOURCE_MENU_OPTIONS.DISCONNECT:
                if (isPipelineRunning) {
                    setIsDisconnectConfirmationDialogVisible(true);
                } else {
                    handleDisconnect();
                }
                break;
            case SOURCE_MENU_OPTIONS.REMOVE:
                handleRemove();
                break;
            case SOURCE_MENU_OPTIONS.EDIT:
                onEdit();
                break;
            case SOURCE_MENU_OPTIONS.TEST:
                void onTest();
                break;
        }
    };

    const handleConnect = () => {
        updatePipeline.mutate(
            {
                params: { path: { project_id } },
                body: { source_id: id },
            },
            {
                onSuccess: () => {
                    toast({
                        type: 'success',
                        message: t('inference.sources.menu.connectSuccess', { name }),
                    });
                },
            }
        );
    };

    const handleDisconnect = () => {
        updatePipeline.mutate(
            {
                params: { path: { project_id } },
                body: { source_id: null },
            },
            {
                onSuccess: () => {
                    toast({
                        type: 'success',
                        message: t('inference.sources.menu.disconnectSuccess', { name }),
                    });
                },
            }
        );
    };

    const handleDisablePipelineAndDisconnect = () => {
        disablePipelineMutation.mutate(
            {
                params: {
                    path: {
                        project_id,
                    },
                },
            },
            {
                onSuccess: () => {
                    updatePipeline.mutate(
                        {
                            params: { path: { project_id } },
                            body: { source_id: null },
                        },
                        {
                            onSuccess: () => {
                                toast({
                                    type: 'success',
                                    message: t('inference.sources.menu.disablePipelineAndDisconnectSuccess', {
                                        name,
                                    }),
                                });

                                setIsDisconnectConfirmationDialogVisible(false);
                            },
                        }
                    );
                },
                onSettled: () => {
                    void stop();
                },
            }
        );
    };

    const handleRemove = () => {
        removeSource.mutate(
            { params: { path: { source_id: id } } },
            {
                onSuccess: () => {
                    toast({
                        type: 'success',
                        message: t('inference.sources.menu.removeSuccess', { name }),
                    });
                },
            }
        );
    };

    return (
        <>
            <MenuTrigger>
                <ActionButton isQuiet aria-label='source menu'>
                    <MoreMenu />
                </ActionButton>
                <Menu
                    onAction={handleOnAction}
                    disabledKeys={isConnected ? [SOURCE_MENU_OPTIONS.REMOVE, SOURCE_MENU_OPTIONS.TEST] : []}
                >
                    {isConnected ? (
                        <Item key={SOURCE_MENU_OPTIONS.DISCONNECT}>{t('inference.sources.menu.disconnect')}</Item>
                    ) : (
                        <Item key={SOURCE_MENU_OPTIONS.CONNECT}>{t('inference.sources.menu.connect')}</Item>
                    )}
                    <Item key={SOURCE_MENU_OPTIONS.TEST}>{t('inference.sources.menu.testConnection')}</Item>
                    <Item key={SOURCE_MENU_OPTIONS.EDIT}>{t('inference.sources.menu.edit')}</Item>
                    <Item key={SOURCE_MENU_OPTIONS.REMOVE}>{t('inference.sources.menu.remove')}</Item>
                </Menu>
            </MenuTrigger>
            <DialogContainer onDismiss={() => setIsDisconnectConfirmationDialogVisible(false)}>
                {isDisconnectConfirmationDialogVisible && (
                    <DisconnectSourceWarningDialog
                        name={name}
                        onDisconnect={handleDisablePipelineAndDisconnect}
                        isPending={disablePipelineMutation.isPending || updatePipeline.isPending}
                        onCancel={() => setIsDisconnectConfirmationDialogVisible(false)}
                    />
                )}
            </DialogContainer>
        </>
    );
};
