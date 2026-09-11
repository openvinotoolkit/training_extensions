// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Key } from 'react';

import { $api } from '@/api';
import { toast } from '@/components/toast/toast.component';
import { useTranslation } from '@/i18n';
import { ActionButton, Item, Menu, MenuTrigger } from '@geti-ui/ui';
import { MoreMenu } from '@geti-ui/ui/icons';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

const SINK_MENU_OPTIONS = {
    CONNECT: 'connect',
    DISCONNECT: 'disconnect',
    REMOVE: 'remove',
    EDIT: 'edit',
    TEST: 'test',
};

export type SinkMenuProps = {
    id: string;
    name: string;
    isConnected: boolean;
    onEdit: () => void;
    onTest: () => Promise<void>;
};

export const SinkMenu = ({ id, name, isConnected, onEdit, onTest }: SinkMenuProps) => {
    const { t } = useTranslation();
    const project_id = useProjectIdentifier();
    const removeSink = $api.useMutation('delete', '/api/sinks/{sink_id}', {
        meta: {
            invalidateQueries: [['get', '/api/sinks']],
        },
    });

    const updatePipeline = $api.useMutation('patch', '/api/projects/{project_id}/pipeline', {
        meta: {
            invalidateQueries: [['get', '/api/projects/{project_id}/pipeline', { params: { path: { project_id } } }]],
        },
    });

    const handleOnAction = (option: Key) => {
        switch (option) {
            case SINK_MENU_OPTIONS.CONNECT:
                handleConnect();
                break;
            case SINK_MENU_OPTIONS.DISCONNECT:
                handleDisconnect();
                break;
            case SINK_MENU_OPTIONS.REMOVE:
                handleRemove();
                break;
            case SINK_MENU_OPTIONS.EDIT:
                onEdit();
                break;
            case SINK_MENU_OPTIONS.TEST:
                void onTest();
                break;
        }
    };

    const handleConnect = () => {
        updatePipeline.mutate(
            {
                params: { path: { project_id } },
                body: { sink_id: id },
            },
            {
                onSuccess: () => {
                    toast({
                        type: 'success',
                        message: t('inference.sinks.menu.connectSuccess', { name }),
                    });
                },
            }
        );
    };

    const handleRemove = () => {
        removeSink.mutate(
            { params: { path: { sink_id: id } } },
            {
                onSuccess: () => {
                    toast({
                        type: 'success',
                        message: t('inference.sinks.menu.removeSuccess', { name }),
                    });
                },
            }
        );
    };

    const handleDisconnect = () => {
        updatePipeline.mutate(
            {
                params: { path: { project_id } },
                body: { sink_id: null },
            },
            {
                onSuccess: () => {
                    toast({
                        type: 'success',
                        message: t('inference.sinks.menu.disconnectSuccess', { name }),
                    });
                },
            }
        );
    };

    return (
        <MenuTrigger>
            <ActionButton isQuiet aria-label='sink menu'>
                <MoreMenu />
            </ActionButton>
            <Menu
                onAction={handleOnAction}
                disabledKeys={isConnected ? [SINK_MENU_OPTIONS.REMOVE, SINK_MENU_OPTIONS.TEST] : []}
            >
                {isConnected ? (
                    <Item key={SINK_MENU_OPTIONS.DISCONNECT}>{t('inference.sinks.menu.disconnect')}</Item>
                ) : (
                    <Item key={SINK_MENU_OPTIONS.CONNECT}>{t('inference.sinks.menu.connect')}</Item>
                )}
                <Item key={SINK_MENU_OPTIONS.TEST}>{t('inference.sinks.menu.testConnection')}</Item>
                <Item key={SINK_MENU_OPTIONS.EDIT}>{t('inference.sinks.menu.edit')}</Item>
                <Item key={SINK_MENU_OPTIONS.REMOVE}>{t('inference.sinks.menu.remove')}</Item>
            </Menu>
        </MenuTrigger>
    );
};
