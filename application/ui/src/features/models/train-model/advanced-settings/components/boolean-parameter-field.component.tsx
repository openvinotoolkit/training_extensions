// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { Switch } from '@geti-ui/ui';

type BooleanParameterFieldProps = {
    value: boolean;
    header: string;
    onChange: (isSelected: boolean) => void;
    isDisabled?: boolean;
};

export const BooleanParameterField = ({ value, header, onChange, isDisabled }: BooleanParameterFieldProps) => {
    const { t } = useTranslation();

    return (
        <Switch
            isEmphasized
            isSelected={value}
            aria-label={`Toggle ${header}`}
            onChange={onChange}
            isDisabled={isDisabled}
        >
            {value ? t('models.training.parameters.on') : t('models.training.parameters.off')}
        </Switch>
    );
};
