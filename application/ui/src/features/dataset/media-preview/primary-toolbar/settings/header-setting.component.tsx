// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ActionButton, Flex, Text } from '@geti-ui/ui';
import { Revisit } from '@geti-ui/ui/icons';
import { useNumberFormatter } from 'react-aria';

export interface HeaderSettingProps {
    headerText: string;
    ariaLabel: string;
    value: number;
    defaultValue: number;
    handleValueChange: (value: number) => void;
    formatOptions: Intl.NumberFormatOptions;
}

export const HeaderSetting = ({
    headerText,
    ariaLabel,
    value,
    handleValueChange,
    formatOptions,
    defaultValue,
}: HeaderSettingProps) => {
    const formatter = useNumberFormatter(formatOptions);

    return (
        <div>
            <Flex alignItems={'center'} justifyContent={'space-between'}>
                <Text>{headerText}</Text>
                <Flex alignItems={'center'} gap={'size-100'} height={'size-400'}>
                    <ActionButton
                        isQuiet
                        onPress={() => handleValueChange(defaultValue)}
                        aria-label={`Reset ${ariaLabel.toLocaleLowerCase()}`}
                    >
                        <Revisit />
                    </ActionButton>

                    <Text>{formatter.format(value)}</Text>
                </Flex>
            </Flex>
        </div>
    );
};
