// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { Flex, Heading, Loading, View } from '@geti-ui/ui';

export const SAMLoading = ({ isLoading }: { isLoading: boolean }) => {
    const { t } = useTranslation();

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
                        {isLoading && t('annotator.tools.autoSegmentation.loading')}
                    </Heading>
                </View>
            </Flex>
        </View>
    );
};
