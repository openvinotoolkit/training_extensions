// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Slider } from '@geti-ui/ui';

import { HeaderSetting, HeaderSettingProps } from './header-setting.component';

export const ImageSetting = ({
    headerText,
    ariaLabel,
    value,
    handleValueChange,
    defaultValue,
    formatOptions,
}: HeaderSettingProps) => {
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
                step={1}
                value={value}
                minValue={-100}
                maxValue={100}
                onChange={handleValueChange}
                aria-label={`${ariaLabel} setting`}
                fillOffset={0}
                isFilled
            />
        </div>
    );
};
