// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useMemo, useState } from 'react';

import { ActionButton, DatePicker, Flex, Text } from '@geti-ui/ui';
import { getLocalTimeZone, now, parseAbsoluteToLocal, type ZonedDateTime } from '@internationalized/date';
import { useDatasetFiltersSearchParams } from 'hooks/use-dataset-filters-search-params.hook';

import classes from './date-filter.module.scss';

const MIN_DATE = parseAbsoluteToLocal(new Date(2020, 0, 30, 0, 0, 0, 0).toISOString());

export const INVALID_RANGE_MESSAGE = 'End date must be later than start date';

const parseDate = (date: string | null): ZonedDateTime | null => (date === null ? null : parseAbsoluteToLocal(date));

const toISOString = (date: ZonedDateTime | null): string | null => (date === null ? null : date.toDate().toISOString());

export const DateFilter = () => {
    const { startDate, endDate, setDateRange } = useDatasetFiltersSearchParams();

    // An empty picker emits a date in the timezone of its placeholder, so this keeps the two pickers
    // and the applied filter on the local timezone instead of a timezone-less date
    const [placeholderValue] = useState(() => now(getLocalTimeZone()).set({ second: 0, millisecond: 0 }));

    // Media cannot be uploaded in the future. The whole current day is allowed so that the bound does
    // not go stale while the filter stays mounted, which would reject the current time.
    const maxDate = useMemo(
        () => placeholderValue.set({ hour: 23, minute: 59, second: 59, millisecond: 999 }),
        [placeholderValue]
    );

    // Range picked by the user whose end date precedes its start date, and thus was not applied as a filter
    const [invalidRange, setInvalidRange] = useState<{ start: ZonedDateTime; end: ZonedDateTime } | null>(null);

    // Memoized so the pickers keep receiving the same value instance while the filter does not change
    const appliedStart = useMemo(() => parseDate(startDate), [startDate]);
    const appliedEnd = useMemo(() => parseDate(endDate), [endDate]);

    const startValue = invalidRange?.start ?? appliedStart;
    const endValue = invalidRange?.end ?? appliedEnd;

    const applyDates = (start: ZonedDateTime | null, end: ZonedDateTime | null) => {
        if (start !== null && end !== null && end.compare(start) < 0) {
            setInvalidRange({ start, end });

            return;
        }

        setInvalidRange(null);

        const newStartDate = toISOString(start);
        const newEndDate = toISOString(end);

        if (newStartDate !== startDate || newEndDate !== endDate) {
            setDateRange(newStartDate, newEndDate);
        }
    };

    const handleStartDateChange = (date: ZonedDateTime | null) => {
        applyDates(date, endValue);
    };

    const handleEndDateChange = (date: ZonedDateTime | null) => {
        applyDates(startValue, date);
    };

    const handleClear = () => {
        setInvalidRange(null);

        if (startDate !== null || endDate !== null) {
            setDateRange(null, null);
        }
    };

    return (
        <Flex direction='column' gap='size-100'>
            <Flex direction='row' alignItems='center' justifyContent='space-between'>
                <Text UNSAFE_className={classes.label}>Filter by upload date</Text>

                <ActionButton
                    isQuiet
                    aria-label='Clear upload date filter'
                    isDisabled={startValue === null && endValue === null}
                    onPress={handleClear}
                >
                    <Text>Clear</Text>
                </ActionButton>
            </Flex>

            <DatePicker
                granularity='minute'
                hourCycle={24}
                hideTimeZone
                width='100%'
                label='Start date'
                labelPosition='top'
                minValue={MIN_DATE}
                maxValue={appliedEnd ?? maxDate}
                placeholderValue={placeholderValue}
                value={startValue}
                onChange={handleStartDateChange}
            />

            <DatePicker
                granularity='minute'
                hourCycle={24}
                hideTimeZone
                width='100%'
                label='End date'
                labelPosition='top'
                minValue={appliedStart ?? MIN_DATE}
                maxValue={maxDate}
                placeholderValue={placeholderValue}
                value={endValue}
                onChange={handleEndDateChange}
                validationState={invalidRange === null ? undefined : 'invalid'}
                errorMessage={invalidRange === null ? undefined : INVALID_RANGE_MESSAGE}
            />
        </Flex>
    );
};
