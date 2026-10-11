// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { ReactNode } from 'react';

import { Link } from '../../platform/components/link.component';

// Everything in this file has legal meaning, so it is kept verbatim in English and never translated.

const ULTRALYTICS_TERMS_URL =
    // eslint-disable-next-line max-len
    'https://www.ultralytics.com/license?utm_source=intel&utm_medium=referral&utm_campaign=geti-model-garden&utm_content=user-notice';

/* eslint-disable max-len */
const INTEL_SIMPLIFIED_NOTICE = [
    'The Geti Windows application is licensed under the Intel Simplified Software License (Version October 2022).',
    'The license sets out the conditions under which you may use and redistribute the application, together with its disclaimers and limitations of liability. Third-party components included with the application may be governed by their own license terms.',
    'Please read the full license terms before continuing.',
].join('\n\n');

const ULTRALYTICS_NOTICE = [
    'Ultralytics YOLO software and models available through Geti are provided by Ultralytics under the GNU Affero General Public License v3.0 ("AGPL-3.0"), unless you have entered into a separate written license agreement with Ultralytics. Accessing, downloading, or retraining these materials through Geti does not grant you an Ultralytics Enterprise License or any other proprietary Ultralytics license.',
    'Depending on how Ultralytics YOLO is used, modified, integrated or made available, AGPL-3.0 may require applicable source code to be made available under AGPL-3.0, including in certain circumstances where software is made available for use over a network. You are responsible for reviewing the AGPL-3.0 terms and determining whether they apply to your intended use. Ultralytics position is that, if you wish to use YOLO in a proprietary commercial product, internal tool or production deployment without making applicable source code available under AGPL-3.0, you should obtain an Ultralytics Enterprise License.',
    'You are responsible for determining which applies to your use. Terms and enterprise licensing options are available at: ',
].join('\n\n');

const DINOV3_NOTICE = [
    'Geti offers classification and segmentation models based on DINOv3, such as EoMT-DINOv3, whose pretrained weights are made available by Meta under the DINOv3 License.',
    'If you train or deploy these models, your use of the DINOv3 weights is governed by the DINOv3 License terms.',
].join('\n\n');

export const LEGAL_STATEMENTS = {
    consent: (licenseName: string) => `By agreeing below, you accept the terms and conditions of the ${licenseName}.`,
    agree: (licenseName: string) => `I have read and agree to the ${licenseName}`,
    optionalInfo: 'This license is provided for your information. It applies only if you use the models it covers.',
};
/* eslint-enable max-len */

export type LicenseNotice = {
    id: string;
    name: string;
    notice: ReactNode;
    href: string;
    isRequired: boolean;
};

export const INTEL_SIMPLIFIED_LICENSE: LicenseNotice = {
    id: 'intel-simplified',
    name: 'Intel Simplified Software License',
    notice: INTEL_SIMPLIFIED_NOTICE,
    // eslint-disable-next-line max-len
    href: 'https://www.intel.com/content/www/us/en/content-details/749362/intel-simplified-software-license-version-october-2022.html',
    isRequired: true,
};

export const ULTRALYTICS_LICENSE: LicenseNotice = {
    id: 'ultralytics',
    name: 'Ultralytics AGPL-3.0 License',
    notice: (
        <>
            {ULTRALYTICS_NOTICE}
            <Link href={ULTRALYTICS_TERMS_URL} target={'_blank'} rel={'noopener noreferrer'}>
                {ULTRALYTICS_TERMS_URL}
            </Link>
        </>
    ),
    href: 'https://github.com/ultralytics/ultralytics/blob/main/LICENSE',
    isRequired: true,
};

export const DINOV3_LICENSE: LicenseNotice = {
    id: 'dinov3',
    name: 'DINOv3 License',
    notice: DINOV3_NOTICE,
    href: 'https://ai.meta.com/resources/models-and-libraries/dinov3-license/',
    isRequired: false,
};
