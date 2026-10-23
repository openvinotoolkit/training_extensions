// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { Text } from '@geti-ui/ui';

import { Toolbar } from '../../shell/toolbar-container/toolbar-container.component';

export const SegmentAnythingOptions = () => {
    const { t } = useTranslation();

    return (
        <Toolbar.Container>
            <Toolbar.Section>
                <Text>{t('annotator.tools.autoSegmentation.hint')}</Text>
            </Toolbar.Section>
        </Toolbar.Container>
    );
};
