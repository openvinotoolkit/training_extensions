// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { useTranslation } from '@/i18n';
import { Button, ButtonGroup, Content, Dialog, Divider, Form, Heading, TextField } from '@geti-ui/ui';

interface RenameModelDialogProps {
    currentName: string;
    onRename: (newName: string) => void;
    onClose: () => void;
    isPending?: boolean;
}

export const RenameModelDialog = ({ currentName, onRename, onClose, isPending }: RenameModelDialogProps) => {
    const { t } = useTranslation();
    const [newName, setNewName] = useState(currentName);

    const hasSameName = newName.trim() === currentName;

    const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();

        onRename(newName.trim());
    };

    return (
        <Dialog>
            <Heading>{t('models.actions.renameDialog.title')}</Heading>

            <Divider />

            <Content>
                <Form onSubmit={handleSubmit} validationBehavior={'native'}>
                    <TextField
                        label={t('models.actions.renameDialog.nameLabel')}
                        value={newName}
                        onChange={setNewName}
                        width='100%'
                        isRequired
                    />
                    <ButtonGroup align={'end'} marginTop={'size-300'}>
                        <Button variant='secondary' onPress={onClose}>
                            {t('models.actions.cancel')}
                        </Button>
                        <Button variant='accent' type='submit' isPending={isPending} isDisabled={hasSameName}>
                            {t('models.actions.rename')}
                        </Button>
                    </ButtonGroup>
                </Form>
            </Content>
        </Dialog>
    );
};
