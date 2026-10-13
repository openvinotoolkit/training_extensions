// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { Button, ButtonGroup, View } from '@geti-ui/ui';

type LicenseFooterProps = {
    isFirst: boolean;
    isLast: boolean;
    canProceed: boolean;
    canAccept: boolean;
    isAccepting: boolean;
    onPrevious: () => void;
    onNext: () => void;
    onAccept: () => void;
};

export const LicenseFooter = ({
    isFirst,
    isLast,
    canProceed,
    canAccept,
    isAccepting,
    onPrevious,
    onNext,
    onAccept,
}: LicenseFooterProps) => {
    const { t } = useTranslation();

    return (
        <View borderTopWidth={'thin'} borderTopColor={'default'} paddingY={'size-200'} paddingX={'size-300'}>
            <ButtonGroup align={'end'} width={'100%'}>
                {!isFirst && (
                    <Button variant={'secondary'} onPress={onPrevious}>
                        {t('license.agreement.previous')}
                    </Button>
                )}
                {isLast ? (
                    <Button variant={'accent'} onPress={onAccept} isPending={isAccepting} isDisabled={!canAccept}>
                        {t('license.agreement.accept')}
                    </Button>
                ) : (
                    <Button variant={'accent'} isDisabled={!canProceed} onPress={onNext}>
                        {t('common.actions.next')}
                    </Button>
                )}
            </ButtonGroup>
        </View>
    );
};
