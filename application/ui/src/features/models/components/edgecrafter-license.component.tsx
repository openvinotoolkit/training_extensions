// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Link } from '../../../platform/components/link.component';

const EDGECRAFTER_LICENSE_URL = 'https://github.com/Intellindust-AI-Lab/EdgeCrafter/blob/main/LICENSE.md';

export const EdgeCrafterLicense = () => {
    return (
        <>
            {'License: '}
            <Link href={EDGECRAFTER_LICENSE_URL} target={'_blank'} rel={'noopener noreferrer'}>
                EdgeCrafter
            </Link>
        </>
    );
};
