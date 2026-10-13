// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { Flex, Grid, Text, View } from '@geti-ui/ui';

import { useAcceptLicense } from './api/use-accept-license.hook';
import { LicenseDetails } from './license-details.component';
import { LicenseFooter } from './license-footer.component';
import { LicenseSteps } from './license-steps.component';
import { LICENSES } from './licenses';
import { useLicenseSteps } from './use-license-steps.hook';

import styles from './license.module.scss';

export const License = () => {
    const { t } = useTranslation();
    const steps = useLicenseSteps(LICENSES);
    const { mutate: acceptLicense, isPending: isAccepting } = useAcceptLicense();

    return (
        <View height={'100vh'} padding={'size-300'} UNSAFE_className={styles.licenseBackground}>
            <Flex justifyContent={'center'} alignItems={'center'} height={'100%'}>
                <View
                    backgroundColor={'gray-50'}
                    borderRadius={'medium'}
                    overflow={'hidden'}
                    width={'100%'}
                    maxWidth={'1000px'}
                    height={'100%'}
                    maxHeight={'660px'}
                    UNSAFE_className={styles.dialog}
                >
                    <Grid columns={['minmax(240px, 30%)', '1fr']} height={'100%'}>
                        <LicenseSteps
                            licenses={LICENSES}
                            currentIndex={steps.currentIndex}
                            agreedIds={steps.agreedIds}
                            lastReachableIndex={steps.lastReachableIndex}
                            onSelect={steps.goTo}
                        />

                        <Flex direction={'column'} minWidth={0} minHeight={0}>
                            <View flex padding={'size-300'} overflow={'auto'} minHeight={0}>
                                <Flex direction={'column'} gap={'size-200'} height={'100%'}>
                                    {LICENSES.length > 1 && (
                                        <Text>
                                            {t('license.agreement.progress', {
                                                current: steps.currentIndex + 1,
                                                total: LICENSES.length,
                                            })}
                                        </Text>
                                    )}
                                    <LicenseDetails
                                        license={steps.current}
                                        isAgreed={steps.isCurrentAgreed}
                                        onAgreedChange={steps.setCurrentAgreed}
                                    />
                                </Flex>
                            </View>

                            <LicenseFooter
                                isFirst={steps.isFirst}
                                isLast={steps.isLast}
                                canProceed={steps.canProceed}
                                canAccept={steps.areAllRequiredAgreed}
                                isAccepting={isAccepting}
                                onPrevious={steps.goToPrevious}
                                onNext={steps.goToNext}
                                onAccept={() => acceptLicense(undefined)}
                            />
                        </Flex>
                    </Grid>
                </View>
            </Flex>
        </View>
    );
};
