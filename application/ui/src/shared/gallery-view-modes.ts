// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ViewModes } from '@geti-ui/ui';

// View modes used by media galleries. ViewModes.DETAILS is intentionally
// excluded because none of the galleries render a details/list view.
export type GalleryViewMode = ViewModes.LARGE | ViewModes.MEDIUM | ViewModes.SMALL;
