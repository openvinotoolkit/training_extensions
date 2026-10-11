// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { Flex, Heading, Text, View } from '@geti-ui/ui';
import { Checkmark, ChevronRight, DocumentIcon } from '@geti-ui/ui/icons';

import type { LicenseNotice } from './license-notices';
import { RequirementBadge } from './requirement-badge.component';

import styles from './license.module.scss';

type LicenseStepProps = {
    license: LicenseNotice;
    index: number;
    isActive: boolean;
    isAgreed: boolean;
    isDisabled: boolean;
    onSelect: (index: number) => void;
};

const LicenseStep = ({ license, index, isActive, isAgreed, isDisabled, onSelect }: LicenseStepProps) => (
    <li>
        <button
            type={'button'}
            className={styles.step}
            aria-current={isActive ? 'step' : undefined}
            disabled={isDisabled}
            onClick={() => onSelect(index)}
        >
            <span className={styles.stepNumber}>{isAgreed ? <Checkmark size={'S'} /> : index + 1}</span>
            <span className={styles.stepLabel}>
                <span>{license.name}</span>
                <RequirementBadge isRequired={license.isRequired} />
            </span>
            <ChevronRight aria-hidden />
        </button>
    </li>
);

type LicenseStepsProps = {
    licenses: LicenseNotice[];
    currentIndex: number;
    agreedIds: ReadonlySet<string>;
    lastReachableIndex: number;
    onSelect: (index: number) => void;
};

export const LicenseSteps = ({
    licenses,
    currentIndex,
    agreedIds,
    lastReachableIndex,
    onSelect,
}: LicenseStepsProps) => {
    const { t } = useTranslation();

    return (
        <View
            elementType={'nav'}
            backgroundColor={'gray-75'}
            borderEndWidth={'thin'}
            borderEndColor={'default'}
            paddingY={'size-300'}
            paddingX={'size-200'}
            overflow={'auto'}
        >
            <Flex direction={'column'} gap={'size-200'}>
                <Flex alignItems={'center'} gap={'size-100'}>
                    <DocumentIcon aria-hidden />
                    <Heading level={2} margin={0}>
                        {t('license.agreement.title')}
                    </Heading>
                </Flex>
                <Text>{t('license.agreement.intro')}</Text>

                <ol className={styles.steps}>
                    {licenses.map((license, index) => (
                        <LicenseStep
                            key={license.id}
                            license={license}
                            index={index}
                            isActive={index === currentIndex}
                            isAgreed={agreedIds.has(license.id)}
                            isDisabled={index > lastReachableIndex}
                            onSelect={onSelect}
                        />
                    ))}
                </ol>
            </Flex>
        </View>
    );
};
