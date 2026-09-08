// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useSearchParams } from 'react-router-dom';
import { render } from 'test-utils/render';

import { END_DATE_PARAM, START_DATE_PARAM } from '../../../../../../hooks/use-dataset-filters-search-params.hook';
import { DateFilter, INVALID_RANGE_MESSAGE } from './date-filter.component';

const START_DATE = '2026-03-15T10:00:00.000Z';
const END_DATE = '2026-03-20T10:00:00.000Z';

const shifted = (date: string, { years = 0, months = 0, days = 0 }) => {
    const local = new Date(date);

    local.setFullYear(local.getFullYear() + years, local.getMonth() + months, local.getDate() + days);

    return local.toISOString();
};

const AppliedFilters = () => {
    const [searchParams] = useSearchParams();

    return (
        <>
            <span data-testid='applied-start-date'>{searchParams.get(START_DATE_PARAM) ?? ''}</span>
            <span data-testid='applied-end-date'>{searchParams.get(END_DATE_PARAM) ?? ''}</span>
        </>
    );
};

const renderDateFilter = (route: string) =>
    render(
        <>
            <DateFilter />
            <AppliedFilters />
        </>,
        { route, path: '/projects/:projectId' }
    );

const setSegment = async (
    user: ReturnType<typeof userEvent.setup>,
    fieldName: string,
    segmentName: RegExp,
    value: string
) => {
    const field = screen.getByRole('group', { name: fieldName });
    const segment = within(field).getByRole('spinbutton', { name: segmentName });

    await user.click(segment);
    await user.keyboard(value);
};

const setYear = (user: ReturnType<typeof userEvent.setup>, fieldName: string, year: string) =>
    setSegment(user, fieldName, /year/i, year);

// A month is committed with a single keystroke, so it does not go through intermediate (applied) values
const setMonth = (user: ReturnType<typeof userEvent.setup>, fieldName: string, month: string) =>
    setSegment(user, fieldName, /month/i, month);

const pickFromCalendar = async (user: ReturnType<typeof userEvent.setup>, index: number, dayLabel: RegExp) => {
    const triggers = screen
        .getAllByRole('button')
        .filter((button) => button.getAttribute('aria-haspopup') === 'dialog');

    await user.click(triggers[index]);

    const calendar = await screen.findByRole('dialog');

    await user.click(within(calendar).getByRole('button', { name: dayLabel }));
};

describe('DateFilter', () => {
    const routeWithDates = `/projects/123?${START_DATE_PARAM}=${START_DATE}&${END_DATE_PARAM}=${END_DATE}`;

    // Without a `placeholderValue` an empty picker emits a date with no timezone, which would be
    // read as UTC and shift the applied filter away from what the user typed
    it('applies a date typed into an empty picker in the local timezone', async () => {
        const user = userEvent.setup();

        renderDateFilter('/projects/123');

        await setSegment(user, 'Start date', /month/i, '3');
        await setSegment(user, 'Start date', /day/i, '15');
        await setSegment(user, 'Start date', /year/i, '2026');
        await setSegment(user, 'Start date', /hour/i, '10');
        await setSegment(user, 'Start date', /minute/i, '30');

        expect(screen.getByTestId('applied-start-date')).toHaveTextContent(new Date(2026, 2, 15, 10, 30).toISOString());
        expect(screen.getByTestId('applied-end-date')).toBeEmptyDOMElement();
    });

    it('applies the filter when the picked range is valid', async () => {
        const user = userEvent.setup();

        renderDateFilter(routeWithDates);

        await setYear(user, 'Start date', '2025');

        expect(screen.getByTestId('applied-start-date')).toHaveTextContent(shifted(START_DATE, { years: -1 }));
        expect(screen.getByTestId('applied-end-date')).toHaveTextContent(END_DATE);
        expect(screen.queryByText(INVALID_RANGE_MESSAGE)).not.toBeInTheDocument();
    });

    // react-spectrum only commits a calendar selection on a field that already holds a value; on an
    // empty one it is deferred until the popover closes, which used to apply it to the other field
    it('applies a date picked from the calendar without changing its time', async () => {
        const user = userEvent.setup();

        renderDateFilter(routeWithDates);

        await pickFromCalendar(user, 0, /march 10, 2026/i);

        expect(screen.getByTestId('applied-start-date')).toHaveTextContent(shifted(START_DATE, { days: -5 }));
        expect(screen.getByTestId('applied-end-date')).toHaveTextContent(END_DATE);
        expect(screen.queryByText(INVALID_RANGE_MESSAGE)).not.toBeInTheDocument();
    });

    it('keeps the previous filter and shows an error when the end date is moved before the start date', async () => {
        const user = userEvent.setup();

        renderDateFilter(routeWithDates);

        await setYear(user, 'End date', '2020');

        expect(screen.getByTestId('applied-start-date')).toHaveTextContent(START_DATE);
        expect(screen.getByTestId('applied-end-date')).toHaveTextContent(END_DATE);
        expect(screen.getByText(INVALID_RANGE_MESSAGE)).toBeVisible();
    });

    it('keeps the previous filter and shows an error when the start date is moved after the end date', async () => {
        const user = userEvent.setup();

        renderDateFilter(routeWithDates);

        await setMonth(user, 'Start date', '4');

        expect(screen.getByTestId('applied-start-date')).toHaveTextContent(START_DATE);
        expect(screen.getByTestId('applied-end-date')).toHaveTextContent(END_DATE);
        expect(screen.getByText(INVALID_RANGE_MESSAGE)).toBeVisible();
    });

    it('applies the filter once an invalid end date is corrected', async () => {
        const user = userEvent.setup();

        renderDateFilter(routeWithDates);

        await setYear(user, 'End date', '2020');
        await setYear(user, 'Start date', '2020');

        expect(screen.getByTestId('applied-start-date')).toHaveTextContent(shifted(START_DATE, { years: -6 }));
        expect(screen.getByTestId('applied-end-date')).toHaveTextContent(shifted(END_DATE, { years: -6 }));
        expect(screen.queryByText(INVALID_RANGE_MESSAGE)).not.toBeInTheDocument();
    });

    it('applies the filter once an invalid start date is corrected', async () => {
        const user = userEvent.setup();

        renderDateFilter(routeWithDates);

        await setMonth(user, 'Start date', '4');
        await setMonth(user, 'Start date', '2');

        expect(screen.getByTestId('applied-start-date')).toHaveTextContent(shifted(START_DATE, { months: -1 }));
        expect(screen.getByTestId('applied-end-date')).toHaveTextContent(END_DATE);
        expect(screen.queryByText(INVALID_RANGE_MESSAGE)).not.toBeInTheDocument();
    });

    it('removes both dates at once', async () => {
        const user = userEvent.setup();

        renderDateFilter(routeWithDates);

        await user.click(screen.getByRole('button', { name: 'Clear upload date filter' }));

        expect(screen.getByTestId('applied-start-date')).toBeEmptyDOMElement();
        expect(screen.getByTestId('applied-end-date')).toBeEmptyDOMElement();
    });

    it('discards an invalid range when the dates are removed', async () => {
        const user = userEvent.setup();

        renderDateFilter(routeWithDates);

        await setYear(user, 'End date', '2020');
        expect(screen.getByText(INVALID_RANGE_MESSAGE)).toBeVisible();

        await user.click(screen.getByRole('button', { name: 'Clear upload date filter' }));

        expect(screen.queryByText(INVALID_RANGE_MESSAGE)).not.toBeInTheDocument();
        expect(screen.getByTestId('applied-start-date')).toBeEmptyDOMElement();
        expect(screen.getByTestId('applied-end-date')).toBeEmptyDOMElement();
    });
});
