// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';

import { Link } from '../../../platform/components/link.component';

export const UltralyticsLicense = () => {
    const { t } = useTranslation();

    return (
        <Link
            href={'https://www.ultralytics.com/legal/agpl-3-0-software-license'}
            target={'_blank'}
            rel={'noopener noreferrer'}
        >
            {t('license.ultralytics.linkText')}
        </Link>
    );
};
