// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { Checkbox, Flex, Heading, Text, Well } from '@geti-ui/ui';
import { LinkOut } from '@geti-ui/ui/icons';

import { Link } from '../../platform/components/link.component';
import { LEGAL_STATEMENTS, type LicenseNotice } from './license-notices';
import { RequirementBadge } from './requirement-badge.component';

import styles from './license.module.scss';

type LicenseDetailsProps = {
    license: LicenseNotice;
    isAgreed: boolean;
    onAgreedChange: (isAgreed: boolean) => void;
};

export const LicenseDetails = ({ license, isAgreed, onAgreedChange }: LicenseDetailsProps) => {
    const { t } = useTranslation();

    return (
        <>
            {/* Spectrum layout components don't forward aria-live. */}
            <div aria-live={'polite'}>
                <Flex direction={'column'} alignItems={'start'} gap={'size-100'}>
                    <Heading level={3} margin={0}>
                        {license.name}
                    </Heading>
                    <RequirementBadge isRequired={license.isRequired} />
                </Flex>
            </div>

            <Text>{license.isRequired ? LEGAL_STATEMENTS.consent(license.name) : LEGAL_STATEMENTS.optionalInfo}</Text>

            <Well
                role={'region'}
                aria-label={'License notice'}
                flex
                minHeight={'size-1200'}
                UNSAFE_className={styles.notice}
            >
                <Flex direction={'column'} alignItems={'start'} gap={'size-200'}>
                    <Text UNSAFE_className={styles.noticeText}>{license.notice}</Text>
                    <Link href={license.href} target={'_blank'} rel={'noopener noreferrer'}>
                        {t('license.agreement.fullLicense')} <LinkOut size={'XS'} />
                    </Link>
                </Flex>
            </Well>

            {license.isRequired && (
                <Checkbox isSelected={isAgreed} onChange={onAgreedChange}>
                    {LEGAL_STATEMENTS.agree(license.name)}
                </Checkbox>
            )}
        </>
    );
};
