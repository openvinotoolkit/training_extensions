// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, Text } from '@geti-ui/ui';

import { pluralizeItems } from '../../../../../../shared/util';

type SelectedMediaCountProps = {
    count: number;
};

export const SelectedMediaCount = ({ count }: SelectedMediaCountProps) => {
    return (
        <Flex direction={'column'} gap={'size-100'}>
            <Text>
                Selected {count} media {pluralizeItems(count)}
            </Text>
        </Flex>
    );
};
