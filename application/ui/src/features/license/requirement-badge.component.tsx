// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { Badge } from '@geti-ui/ui';

export const RequirementBadge = ({ isRequired }: { isRequired: boolean }) => {
    const { t } = useTranslation();

    return (
        <Badge variant={isRequired ? 'info' : 'neutral'}>
            {isRequired ? t('license.agreement.required') : t('license.agreement.optional')}
        </Badge>
    );
};
