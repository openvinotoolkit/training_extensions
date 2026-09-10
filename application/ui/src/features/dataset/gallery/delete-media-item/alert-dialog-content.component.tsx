// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { AlertDialog, Text } from '@geti-ui/ui';

type AlertDialogContentProps = {
    itemsIds: string[];
    onPrimaryAction: () => void;
};

export const AlertDialogContent = ({ itemsIds, onPrimaryAction }: AlertDialogContentProps) => {
    const { t } = useTranslation();

    return (
        <AlertDialog
            maxHeight={'size-6000'}
            title={t('dataset.delete.title')}
            variant='destructive'
            primaryActionLabel={t('dataset.delete.confirm')}
            secondaryActionLabel={t('common.actions.cancel')}
            onPrimaryAction={onPrimaryAction}
            autoFocusButton='primary'
        >
            <Text>{t('dataset.delete.confirmation', { count: itemsIds.length })}</Text>
        </AlertDialog>
    );
};
