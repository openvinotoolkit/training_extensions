// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, Heading, Loading, View } from '@geti-ui/ui';

export const ToolLoading = ({ message }: { message: string }) => {
    return (
        <View
            position={'absolute'}
            left={0}
            top={0}
            right={0}
            bottom={0}
            UNSAFE_style={{
                backgroundColor: 'var(--spectrum-alias-background-color-modal-overlay)',
                zIndex: 10,
            }}
        >
            <Flex direction={'column'} alignItems={'center'} justifyContent={'center'} height='100%' gap='size-100'>
                <View
                    UNSAFE_style={{
                        transform: 'scale(calc(1 / var(--zoom-scale, 1)))',
                        transformOrigin: 'center',
                    }}
                >
                    <Loading mode='inline' height={'auto'} variant='intel' />
                    <Heading
                        level={3}
                        UNSAFE_style={{
                            textShadow: '1px 1px 2px black, 1px 1px 2px white',
                        }}
                    >
                        {message}
                    </Heading>
                </View>
            </Flex>
        </View>
    );
};
