// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

const VALID_VIDEO_EXT = ['mp4', 'avi', 'mkv', 'mov', 'webm', 'm4v'];
const VALID_IMAGE_EXT = ['jpg', 'jpeg', 'png', 'jfif', 'tif', 'tiff', 'webp', 'bmp'];
const VALID_EXT = [...VALID_VIDEO_EXT, ...VALID_IMAGE_EXT];

export const acceptedExtensions = VALID_EXT.map((ext) => `.${ext}`).join(',');
export const acceptedVideoExtensions = VALID_VIDEO_EXT.map((ext) => `.${ext}`).join(',');

const getFileExtension = (file: File): string => file.name.split('.').pop()?.toLowerCase() ?? '';

export const isVideoFile = (file: File) => VALID_VIDEO_EXT.includes(getFileExtension(file));
