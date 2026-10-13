// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { getMockedLogEntry, getMockedLogEntryJson } from 'mocks/mock-log-entry';

import { filterLogs, parseLogLine } from './log-utils';

describe('parseLogLine', () => {
    it('parses a valid JSON log line into a LogEntry', () => {
        const line = getMockedLogEntryJson({ message: 'Starting training' });
        const entry = parseLogLine(line);

        expect(entry).not.toBeNull();
        expect(entry?.record.message).toBe('Starting training');
        expect(entry?.record.level.name).toBe('INFO');
    });

    it('returns a plain text entry for non-JSON lines', () => {
        const entry = parseLogLine('Some plain text output');

        expect(entry).not.toBeNull();
        expect(entry?.record.message).toBe('Some plain text output');
        expect(entry?.record.level.name).toBe('INFO');
        expect(entry?.record.module).toBe('');
    });

    it('returns a plain text entry for JSON that does not match LogEntry shape', () => {
        const entry = parseLogLine('{"key": "value"}');

        expect(entry).not.toBeNull();
        expect(entry?.record.message).toBe('{"key": "value"}');
    });

    it('returns null for empty strings', () => {
        expect(parseLogLine('')).toBeNull();
        expect(parseLogLine('   ')).toBeNull();
    });
});

describe('filterLogs', () => {
    const mockLogs = [
        getMockedLogEntry({ level: { icon: '🐛', name: 'DEBUG', no: 10 }, message: 'Debug message' }),
        getMockedLogEntry({ level: { icon: 'ℹ️', name: 'INFO', no: 20 }, message: 'Starting training loop' }),
        getMockedLogEntry({ level: { icon: 'ℹ️', name: 'INFO', no: 20 }, message: 'Epoch 1/30' }),
        getMockedLogEntry({ level: { icon: '⚠️', name: 'WARNING', no: 30 }, message: 'Low GPU memory detected' }),
        getMockedLogEntry({
            level: { icon: '❌', name: 'ERROR', no: 40 },
            message: 'Validation metric below threshold',
        }),
        getMockedLogEntry({
            level: { icon: '✅', name: 'SUCCESS', no: 25 },
            message: 'Training completed successfully',
        }),
        getMockedLogEntry({ level: { icon: '🔥', name: 'CRITICAL', no: 50 }, message: 'Out of memory' }),
    ];

    it('filters out logs below minimum level', () => {
        const result = filterLogs(mockLogs, 'WARNING', '');

        expect(result).toHaveLength(3);
        expect(result.map((log) => log.record.level.name)).toEqual(['WARNING', 'ERROR', 'CRITICAL']);
    });

    it('filters logs by case-insensitive search query', () => {
        const result = filterLogs(mockLogs, 'DEBUG', 'training');

        expect(result).toHaveLength(2);
        expect(result.map((log) => log.record.message)).toEqual([
            'Starting training loop',
            'Training completed successfully',
        ]);
    });

    it('applies both level and search filters together', () => {
        const result = filterLogs(mockLogs, 'WARNING', 'memory');

        expect(result).toHaveLength(2);
        expect(result.map((log) => log.record.level.name)).toEqual(['WARNING', 'CRITICAL']);
    });
});
