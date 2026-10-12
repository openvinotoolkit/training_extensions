// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import getiLogo from '@/assets/icons/geti-logo.webp';
import { useTranslation } from '@/i18n';
import { Flex, Text } from '@geti-ui/ui';
import { Link } from 'react-router';

import { paths } from '../../constants/paths';

import classes from './geti-logo-link.module.scss';

export const GetiLogoLink = () => {
    const { t } = useTranslation();

    return (
        <Link to={paths.project.index({})} viewTransition>
            <Flex alignItems={'center'} gap={'size-50'}>
                <img src={getiLogo} alt={t('navigation.logoAlt')} className={classes.logo} />
                <Text UNSAFE_className={classes.logoText}>Geti™</Text>
            </Flex>
        </Link>
    );
};
