// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Loguru serialized log format (serialize=True)
// https://loguru.readthedocs.io/en/stable/api/logger.html#record

export type LogLevel = 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR' | 'SUCCESS' | 'CRITICAL';

export type LogRecord = {
    elapsed: { repr: string; seconds: number };
    exception: { type: string; value: string; traceback: boolean } | null;
    extra: Record<string, unknown>;
    file: { name: string; path: string };
    function: string;
    level: { icon: string; name: LogLevel; no: number };
    line: number;
    message: string;
    module: string;
    name: string;
    process: { id: number; name: string };
    thread: { id: number; name: string };
    time: { repr: string; timestamp: number };
};

export type LogEntry = {
    text: string;
    record: LogRecord;
};

export const LOG_LEVEL_COLORS: Record<LogLevel, string> = {
    DEBUG: 'var(--spectrum-global-color-gray-600)',
    INFO: 'var(--spectrum-global-color-gray-900)',
    WARNING: 'var(--spectrum-global-color-orange-600)',
    ERROR: 'var(--spectrum-global-color-red-600)',
    SUCCESS: 'var(--spectrum-global-color-green-600)',
    CRITICAL: 'var(--spectrum-global-color-magenta-600)',
};

export const LOG_LEVEL_PRIORITY: Record<LogLevel, number> = {
    DEBUG: 10,
    INFO: 20,
    SUCCESS: 25,
    WARNING: 30,
    ERROR: 40,
    CRITICAL: 50,
};
