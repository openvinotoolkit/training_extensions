// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { Link } from '@geti-ui/ui';

type ParentRevisionModelProps = {
    id: string | undefined;
    name: string;
    onExpandModel?: (id: string) => void;
};

export const ParentRevisionModel = ({ id, name, onExpandModel }: ParentRevisionModelProps) => {
    const { t } = useTranslation();

    return (
        <>
            {t('models.list.fineTunedFrom')}{' '}
            <Link
                UNSAFE_style={{ textDecoration: 'none' }}
                onPress={() => {
                    if (id) {
                        onExpandModel?.(id);
                    }
                }}
            >
                {name}
            </Link>
        </>
    );
};
