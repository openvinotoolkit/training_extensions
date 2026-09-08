// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { isEmpty, isString } from 'lodash-es';
import prettyBytes from 'pretty-bytes';

const pluralRules = new Intl.PluralRules('en');

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type GetElementType<T extends any[]> = T extends (infer U)[] ? U : never;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type IsValidArrayType<T> = T extends any[] ? GetElementType<T> : never;
export const isNonEmptyArray = <T>(value: T): value is IsValidArrayType<T> => Array.isArray(value) && !isEmpty(value);

export const isNonEmptyString = (value: unknown): value is string => isString(value) && value !== '';

export const formatBytes = (bytes: number): string => prettyBytes(bytes);

export const pluralize = (count: number, singular: string, plural: string): string =>
    pluralRules.select(count) === 'one' ? singular : plural;

export const pluralizeItems = (count: number): string => pluralize(count, 'item', 'items');

export function assertIsNotNullable<T>(value: T | null | undefined, name = 'value'): asserts value is T {
    if (value === null || value === undefined) {
        throw new Error(`${name} must not be null or undefined`);
    }
}
