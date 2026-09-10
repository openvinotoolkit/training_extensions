// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { AlertDialog, DialogContainer } from '@geti-ui/ui';

type EnablePipelineBlockedDialogProps = {
    isOpen: boolean;
    onClose: () => void;
};

export const EnablePipelineBlockedDialog = ({ isOpen, onClose }: EnablePipelineBlockedDialogProps) => {
    const { t } = useTranslation();

    return (
        <DialogContainer onDismiss={onClose}>
            {isOpen && (
                <AlertDialog
                    title={t('project.panel.enablePipelineBlocked.title')}
                    primaryActionLabel={t('project.panel.enablePipelineBlocked.close')}
                    variant={'warning'}
                    onPrimaryAction={onClose}
                >
                    {t('project.panel.enablePipelineBlocked.message')}
                </AlertDialog>
            )}
        </DialogContainer>
    );
};
