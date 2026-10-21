// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import type { Label } from '@/api/types';
import { fireEvent, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { getMockedLabel } from 'mocks/mock-labels';
import { render } from 'test-utils/render';

import { EMPTY_LABEL_ID } from '../../../../../shared/labels';
import { LabelsList } from './labels-list.component';

const labels: Label[] = [
    getMockedLabel({ id: 'id-cat', name: 'Cat' }),
    getMockedLabel({ id: 'id-dog', name: 'Dog' }),
    getMockedLabel({ id: 'id-bird', name: 'Bird' }),
];

const emptyLabel: Label = getMockedLabel({ id: EMPTY_LABEL_ID, name: 'No label' });

const ControlledLabelsList = ({
    initialSelected = new Set<string>(),
    labels: labelsProp = labels,
    isMultiple = false,
    onSelectedLabelsChange = rs.fn(),
}: {
    initialSelected?: Set<string>;
    labels?: Label[];
    isMultiple?: boolean;
    onSelectedLabelsChange?: (selected: Set<string>) => void;
}) => {
    const [selected, setSelected] = useState<Set<string>>(initialSelected);

    const handleChange = (next: Set<string>) => {
        setSelected(next);
        onSelectedLabelsChange(next);
    };

    return (
        <LabelsList
            ariaLabel='Test labels'
            labels={labelsProp}
            selectedLabels={selected}
            onSelectedLabelsChange={handleChange}
            isMultiple={isMultiple}
        />
    );
};

const getLabelItem = (name: string) => screen.queryByRole('checkbox', { name: `Select ${name}` });

const getSearchInput = () => screen.getByRole('textbox', { name: 'Search labels' });

const searchByLabelName = async (name: string) => {
    await userEvent.type(getSearchInput(), name);
};

const clearSearch = async () => {
    await userEvent.clear(getSearchInput());
};

describe('LabelsList', () => {
    describe('search / filtering', () => {
        it('filters labels based on search phrase (case-insensitive)', async () => {
            render(<ControlledLabelsList />);

            await searchByLabelName('cat');

            expect(getLabelItem('Cat')).toBeInTheDocument();
            expect(getLabelItem('Dog')).not.toBeInTheDocument();
            expect(getLabelItem('Bird')).not.toBeInTheDocument();
        });

        it('is case-insensitive when filtering', async () => {
            render(<ControlledLabelsList />);

            await searchByLabelName('CAT');

            expect(getLabelItem('Cat')).toBeInTheDocument();
            expect(getLabelItem('Dog')).not.toBeInTheDocument();
            expect(getLabelItem('Bird')).not.toBeInTheDocument();
        });

        it('shows "No results found" message when search has no matches', async () => {
            render(<ControlledLabelsList />);

            await searchByLabelName('xyz-abc');

            expect(screen.getByText(/no results found/i)).toBeInTheDocument();
            expect(getLabelItem('Cat')).not.toBeInTheDocument();
            expect(getLabelItem('Dog')).not.toBeInTheDocument();
            expect(getLabelItem('Bird')).not.toBeInTheDocument();
        });

        it('shows all labels again after clearing search phrase', async () => {
            render(<ControlledLabelsList />);

            await searchByLabelName('cat');
            await clearSearch();

            expect(getLabelItem('Cat')).toBeInTheDocument();
            expect(getLabelItem('Dog')).toBeInTheDocument();
            expect(getLabelItem('Bird')).toBeInTheDocument();
        });

        it('does not show "No results found" for an empty search phrase', () => {
            render(<ControlledLabelsList />);

            expect(screen.queryByText(/no results found/i)).not.toBeInTheDocument();
        });

        it('shows multiple matching results', async () => {
            const multipleLabels: Label[] = [
                getMockedLabel({ id: 'id-1', name: 'Cat' }),
                getMockedLabel({ id: 'id-2', name: 'Category' }),
                getMockedLabel({ id: 'id-3', name: 'Dog' }),
            ];

            render(<ControlledLabelsList labels={multipleLabels} />);

            await searchByLabelName('cat');

            expect(getLabelItem('Cat')).toBeInTheDocument();
            expect(getLabelItem('Category')).toBeInTheDocument();
            expect(getLabelItem('Dog')).not.toBeInTheDocument();
        });
    });

    describe('single-select mode', () => {
        it('calls onSelectedLabelsChange with the clicked label id', () => {
            const onSelectedLabelsChange = rs.fn();
            render(<ControlledLabelsList isMultiple={false} onSelectedLabelsChange={onSelectedLabelsChange} />);

            const catLabel = getLabelItem('Cat');
            catLabel && fireEvent.click(catLabel);

            expect(onSelectedLabelsChange).toHaveBeenCalledWith(new Set(['id-cat']));

            const dogLabel = getLabelItem('Dog');
            dogLabel && fireEvent.click(dogLabel);

            expect(onSelectedLabelsChange).toHaveBeenCalledWith(new Set(['id-dog']));
        });

        it('marks the selected label', async () => {
            render(<ControlledLabelsList isMultiple={false} initialSelected={new Set(['id-cat'])} />);

            expect(getLabelItem('Cat')).toBeChecked();
            expect(getLabelItem('Dog')).not.toBeChecked();
        });
    });

    describe('multi-select mode', () => {
        it('allows selecting multiple labels', () => {
            const onSelectedLabelsChange = rs.fn();
            render(<ControlledLabelsList isMultiple onSelectedLabelsChange={onSelectedLabelsChange} />);

            const catLabel = getLabelItem('Cat');
            catLabel && fireEvent.click(catLabel);

            const dogLabel = getLabelItem('Dog');
            dogLabel && fireEvent.click(dogLabel);

            expect(onSelectedLabelsChange).toHaveBeenLastCalledWith(new Set(['id-cat', 'id-dog']));
        });

        it('marks all selected labels ', () => {
            render(<ControlledLabelsList isMultiple initialSelected={new Set(['id-cat', 'id-dog'])} />);

            expect(getLabelItem('Cat')).toBeChecked();
            expect(getLabelItem('Dog')).toBeChecked();
            expect(getLabelItem('Bird')).not.toBeChecked();
        });
    });

    describe('empty label selection logic', () => {
        it('selects only the empty label when it is selected and no other label was already selected', () => {
            const onSelectedLabelsChange = rs.fn();
            render(
                <ControlledLabelsList
                    isMultiple
                    labels={[...labels, emptyLabel]}
                    onSelectedLabelsChange={onSelectedLabelsChange}
                />
            );

            const noLabel = getLabelItem('No label');

            noLabel && fireEvent.click(noLabel);

            expect(onSelectedLabelsChange).toHaveBeenCalledWith(new Set([EMPTY_LABEL_ID]));
        });

        it('deselects the empty label when a regular label is added to an empty-label-only selection', () => {
            const onSelectedLabelsChange = rs.fn();
            render(
                <ControlledLabelsList
                    isMultiple
                    labels={[...labels, emptyLabel]}
                    initialSelected={new Set([EMPTY_LABEL_ID])}
                    onSelectedLabelsChange={onSelectedLabelsChange}
                />
            );

            const catLabel = getLabelItem('Cat');

            catLabel && fireEvent.click(catLabel);

            expect(onSelectedLabelsChange).toHaveBeenCalledWith(new Set(['id-cat']));
        });

        it('deselects non-empty labels when the empty label gets selected', () => {
            const onSelectedLabelsChange = rs.fn();

            render(
                <ControlledLabelsList
                    isMultiple
                    labels={[...labels, emptyLabel]}
                    initialSelected={new Set(['id-cat', 'id-dog'])}
                    onSelectedLabelsChange={onSelectedLabelsChange}
                />
            );

            const noLabel = getLabelItem('No label');

            noLabel && fireEvent.click(noLabel);

            expect(onSelectedLabelsChange).toHaveBeenCalledWith(new Set([EMPTY_LABEL_ID]));
        });
    });
});
