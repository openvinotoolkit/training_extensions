// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

const pad = (value: number) => String(value).padStart(2, '0');

export const formatCompactDuration = (seconds: number) => {
    const totalSeconds = Math.floor(seconds);
    const hours = Math.floor(totalSeconds / 3600);
    const minutesAndSeconds = `${pad(Math.floor((totalSeconds % 3600) / 60))}:${pad(totalSeconds % 60)}`;

    return hours === 0 ? minutesAndSeconds : `${pad(hours)}:${minutesAndSeconds}`;
};
