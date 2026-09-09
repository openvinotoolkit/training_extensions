// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Fragment } from 'react';

import dayjs from 'dayjs';
import { useClipboard } from 'hooks/use-clipboard/use-clipboard.hook';

import { LOG_LEVEL_COLORS, type LogEntry as LogEntryType } from './log-types';

import classes from './log-entry.module.scss';

type LogEntryProps = {
    entry: LogEntryType;
};

// Matches URLs (https?://...), absolute paths (/foo, C:\foo), and relative multi-segment paths (foo/bar/baz)
const PATH_REGEX = /(https?:\/\/[^\s'"<>]+|(?:\/|[A-Za-z]:\\)[^\s'"<>]+|[\w][\w.-]*(?:\/[\w.-]+)+)/g;

const formatTimestamp = (timestamp: number): string => {
    if (!timestamp) {
        return '';
    }

    return dayjs.unix(timestamp).format('HH:mm:ss');
};

const formatSource = (name: string, func: string, line: number): string => {
    if (!name && !func) {
        return '';
    }

    const parts = [name || '', func || ''];

    if (line > 0) {
        parts.push(String(line));
    }

    return parts.filter(Boolean).join(':');
};

const MessageWithPaths = ({ message }: { message: string }) => {
    const { copy } = useClipboard();

    return message.split(PATH_REGEX).map((part, index) => {
        if (index % 2 === 1) {
            return (
                <span
                    key={index}
                    className={classes.path}
                    title={'Click to copy path'}
                    onClick={() => copy(part)}
                    onKeyDown={(event) => {
                        if (event.repeat) {
                            return;
                        }

                        if (event.key === 'Enter' || event.key === ' ') {
                            event.preventDefault();
                            copy(part);
                        }
                    }}
                    role={'button'}
                    tabIndex={0}
                >
                    {part}
                </span>
            );
        }

        return <Fragment key={index}>{part}</Fragment>;
    });
};

export const LogEntry = ({ entry }: LogEntryProps) => {
    const { record } = entry;
    const levelColor = LOG_LEVEL_COLORS[record.level.name] ?? LOG_LEVEL_COLORS.INFO;
    const timestamp = formatTimestamp(record.time.timestamp);
    const source = formatSource(record.name, record.function, record.line);

    return (
        <div className={classes.logEntry} role={'listitem'}>
            {timestamp ? <span className={classes.timestamp}>{timestamp}</span> : null}
            <span className={classes.level} style={{ color: levelColor }}>
                {record.level.name}
            </span>
            {source ? <span className={classes.source}>{source}</span> : null}
            <span className={classes.message}>
                <MessageWithPaths message={record.message.trim()} />
            </span>
        </div>
    );
};
