// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { clsx } from 'clsx';

import classes from './icon-wrapper.module.scss';

export const IconWrapper = ({
    children,
    isSelected,
    isDisabled,
}: {
    children: ReactNode;
    isSelected?: boolean;
    isDisabled?: boolean;
}) => {
    return (
        <div className={clsx(classes.iconWrapper, { [classes.selected]: isSelected, [classes.disabled]: isDisabled })}>
            {children}
        </div>
    );
};
