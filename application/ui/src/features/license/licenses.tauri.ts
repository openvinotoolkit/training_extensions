// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { INTEL_SIMPLIFIED_LICENSE, ULTRALYTICS_LICENSE, type LicenseNotice } from './license-notices';

// The desktop (MSIX) app is itself distributed under the Intel Simplified Software License.
export const LICENSES: LicenseNotice[] = [INTEL_SIMPLIFIED_LICENSE, ULTRALYTICS_LICENSE /*DINOV3_LICENSE*/];
