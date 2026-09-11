// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { SortDown, SortUp, SortUpDown } from '@geti-ui/ui/icons';
import { clsx } from 'clsx';

import type { SortBy, SortDescriptor, SortDirection } from '../types';
import { DEFAULT_SORT_DIRECTIONS } from '../utils/sorting';

import classes from './column-header.module.scss';

type ColumnHeaderProps = {
    label: string;
    ariaLabel?: string;
    sortKey: SortBy;
    sortBy: SortDescriptor;
    onSortChange: (key: SortBy) => void;
};

const SORT_ICONS = { asc: SortUp, desc: SortDown };

const DIRECTION_LABELS: Record<SortDirection, string> = { asc: 'ascending', desc: 'descending' };

export const ColumnHeader = ({ label, ariaLabel, sortKey, sortBy, onSortChange }: ColumnHeaderProps) => {
    const isSorted = sortBy.key === sortKey;
    const direction = isSorted ? sortBy.direction : DEFAULT_SORT_DIRECTIONS[sortKey];
    const SortIcon = isSorted ? SORT_ICONS[direction] : SortUpDown;
    const accessibleLabel = ariaLabel ?? label;

    return (
        <button
            className={clsx(classes.columnHeader, { [classes.isSorted]: isSorted })}
            onClick={() => onSortChange(sortKey)}
            aria-label={
                isSorted ? `${accessibleLabel}, sorted ${DIRECTION_LABELS[direction]}` : `Sort by ${accessibleLabel}`
            }
        >
            {label}
            <span className={classes.icon} aria-hidden={'true'}>
                <SortIcon width={16} height={16} />
            </span>
        </button>
    );
};
