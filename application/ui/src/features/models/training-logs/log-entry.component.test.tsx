// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, render, screen } from '@testing-library/react';
import { useClipboard } from 'hooks/use-clipboard/use-clipboard.hook';
import { getMockedLogEntry } from 'mocks/mock-log-entry';

import { LogEntry } from './log-entry.component';
import { LogRecord } from './log-types';

const mockCopy = vi.fn();

vi.mock('hooks/use-clipboard/use-clipboard.hook', () => ({
    useClipboard: () => ({ copy: mockCopy }),
}));

vi.mocked(useClipboard);

const renderLogEntry = (overrides: Partial<LogRecord> = {}) => {
    const entry = getMockedLogEntry(overrides);

    return render(<LogEntry entry={entry} />);
};

describe('LogEntry', () => {
    beforeEach(() => {
        mockCopy.mockClear();
    });

    describe('path regex detection in messages', () => {
        it('renders a https URL as a clickable span', () => {
            renderLogEntry({ message: 'See https://example.com/docs for details' });
            expect(screen.getByText('https://example.com/docs')).toBeInTheDocument();
            expect(screen.getByTitle('Click to copy path')).toBeInTheDocument();
        });

        it('renders a http URL as a clickable span', () => {
            renderLogEntry({ message: 'Visit http://localhost:7860/api' });
            expect(screen.getByText('http://localhost:7860/api')).toBeInTheDocument();
            expect(screen.getByTitle('Click to copy path')).toBeInTheDocument();
        });

        it('renders an absolute Unix path as a clickable span', () => {
            renderLogEntry({ message: 'File saved at /home/user/output/model.pt' });

            expect(screen.getByText('/home/user/output/model.pt')).toBeInTheDocument();
            expect(screen.getByTitle('Click to copy path')).toBeInTheDocument();
        });

        it('renders a Windows absolute path as a clickable span', () => {
            renderLogEntry({ message: 'Exported to C:\\Users\\user\\model.xml' });

            expect(screen.getByText('C:\\Users\\user\\model.xml')).toBeInTheDocument();
            expect(screen.getByTitle('Click to copy path')).toBeInTheDocument();
        });

        it('renders a relative multi-segment path as a clickable span', () => {
            renderLogEntry({ message: 'Loading config/train/default.yaml' });

            expect(screen.getByText('config/train/default.yaml')).toBeInTheDocument();
            expect(screen.getByTitle('Click to copy path')).toBeInTheDocument();
        });

        it('renders multiple paths in a single message', () => {
            renderLogEntry({ message: 'Copied /src/model.py to /dst/model.py', name: '', function: '' });

            expect(screen.getAllByTitle('Click to copy path')).toHaveLength(2);
            expect(screen.getByText('/src/model.py')).toBeInTheDocument();
            expect(screen.getByText('/dst/model.py')).toBeInTheDocument();
        });

        it('does not mark plain text without slashes as a path', () => {
            renderLogEntry({ message: 'Training completed successfully', name: '', function: '' });

            expect(screen.queryByTitle('Click to copy path')).not.toBeInTheDocument();
        });

        it('does not treat a single-segment word as a path', () => {
            renderLogEntry({ message: 'justoneword', name: '', function: '' });

            expect(screen.queryByTitle('Click to copy path')).not.toBeInTheDocument();
        });

        it('copies the path to clipboard when the span is clicked', () => {
            renderLogEntry({ message: 'Saved to /tmp/output/result.json', name: '', function: '' });

            fireEvent.click(screen.getByTitle('Click to copy path'));

            expect(mockCopy).toHaveBeenCalledWith('/tmp/output/result.json');
        });
    });

    describe('exception traceback rendering', () => {
        it('renders the traceback appended in `text` when the record has an exception', () => {
            const entry = getMockedLogEntry({
                message: 'Unhandled exception in worker',
                exception: { type: 'ValueError', value: 'bad value', traceback: true },
            });
            entry.text =
                '2026-09-10 10:00:00 | ERROR | mod:fn:1 - Unhandled exception in worker\n' +
                'Traceback (most recent call last):\n  raise ValueError("bad value")\nValueError: bad value';

            render(<LogEntry entry={entry} />);

            expect(screen.getByText(/Traceback \(most recent call last\)/)).toBeInTheDocument();
            expect(screen.getByText(/ValueError: bad value/)).toBeInTheDocument();
        });

        it('falls back to record.message when there is no exception', () => {
            const entry = getMockedLogEntry({ message: 'Plain message' });
            entry.text = 'unrelated formatted text';

            render(<LogEntry entry={entry} />);

            expect(screen.getByText('Plain message')).toBeInTheDocument();
        });
    });
});
