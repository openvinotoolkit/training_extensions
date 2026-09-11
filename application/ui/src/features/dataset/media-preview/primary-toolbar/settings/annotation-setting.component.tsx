// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Slider } from '@geti-ui/ui';

import { HeaderSetting, HeaderSettingProps } from './header-setting.component';

interface AnnotationSettingsProps extends HeaderSettingProps {
    isDisabled?: boolean;
}

export const AnnotationSetting = ({
    headerText,
    ariaLabel,
    value,
    handleValueChange,
    formatOptions,
    defaultValue,
    isDisabled,
}: AnnotationSettingsProps) => {
    return (
        <div aria-label={ariaLabel}>
            <HeaderSetting
                headerText={headerText}
                ariaLabel={ariaLabel}
                value={value}
                defaultValue={defaultValue}
                formatOptions={formatOptions}
                handleValueChange={handleValueChange}
            />
            <Slider
                width={'100%'}
                minValue={0}
                maxValue={1}
                step={0.01}
                value={value}
                onChange={handleValueChange}
                aria-label={`${ariaLabel} setting`}
                isDisabled={isDisabled}
                isFilled
            />
        </div>
    );
};
