// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { Button, Content, Divider, Flex, Heading, Text, View } from '@geti-ui/ui';

import { Link } from '../../platform/components/link.component';
import { useAcceptLicense } from './api/use-accept-license.hook';

import styles from './license.module.scss';

const LICENSE_LINKS = {
    intelSimplified: {
        // eslint-disable-next-line max-len
        href: 'https://www.intel.com/content/www/us/en/content-details/749362/intel-simplified-software-license-version-october-2022.html',
    },

    dinov2: {
        href: 'https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md',
    },
};

export const License = () => {
    const { t } = useTranslation();
    const { mutate: acceptLicense, isPending: isAccepting } = useAcceptLicense();

    return (
        <View UNSAFE_className={styles.licenseBackground} height={'100vh'}>
            <Flex justifyContent={'center'} alignItems={'center'} height={'100%'}>
                <View
                    backgroundColor={'gray-50'}
                    padding={'size-400'}
                    borderRadius={'regular'}
                    maxWidth={'size-6000'}
                    width={'100%'}
                >
                    <Heading level={2}>{t('license.agreement.title')}</Heading>
                    <Divider marginY={'size-200'} size={'S'} />
                    <Content>
                        <Text>{t('license.agreement.intro')}</Text>
                        <ul className={styles.list}>
                            <li>{t('license.agreement.termsRead')}</li>
                            <li>{t('license.agreement.termsGovern')}</li>
                            <li>{t('license.agreement.termsAccepted')}</li>
                        </ul>
                        <Flex direction={'column'} marginTop={'size-200'}>
                            <Link
                                href={LICENSE_LINKS.intelSimplified.href}
                                target={'_blank'}
                                rel={'noopener noreferrer'}
                            >
                                {t('license.links.intelSimplified')}
                            </Link>
                            <Link href={LICENSE_LINKS.dinov2.href} target={'_blank'} rel={'noopener noreferrer'}>
                                {t('license.links.dinov2')}
                            </Link>
                        </Flex>
                    </Content>
                    <Flex justifyContent={'end'} marginTop={'size-300'}>
                        <Button
                            variant={'accent'}
                            onPress={() => acceptLicense(undefined)}
                            isPending={isAccepting}
                            isDisabled={isAccepting}
                        >
                            {t('license.agreement.accept')}
                        </Button>
                    </Flex>
                </View>
            </Flex>
        </View>
    );
};
