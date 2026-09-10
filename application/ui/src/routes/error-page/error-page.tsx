// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { Button, Heading, IllustratedMessage, View } from '@geti-ui/ui';
import { NotFound } from '@geti-ui/ui/icons';
import { isObject, isString } from 'lodash-es';
import { isRouteErrorResponse, useRouteError } from 'react-router-dom';

import { paths } from '../../constants/paths';
import { redirectTo } from '../utils';

const useErrorMessage = () => {
    const { t } = useTranslation();
    const error = useRouteError();

    if (isRouteErrorResponse(error)) {
        if (error.status === 400) {
            return t('application.errorPage.badRequest');
        }

        if (error.status === 403) {
            return t('application.errorPage.forbidden');
        }

        if (error.status === 404) {
            return t('application.errorPage.notFound');
        }

        if (error.status === 401) {
            return t('application.errorPage.unauthorized');
        }

        if (error.status === 500) {
            return t('application.errorPage.serverError');
        }

        if (error.status === 503) {
            return t('application.errorPage.serviceUnavailable');
        }
    }

    if (error instanceof TypeError) {
        return error.message;
    }

    if (isObject(error) && 'detail' in error && isString(error.detail)) {
        return error.detail;
    }

    return t('application.errorPage.unknownError');
};

export const ErrorPage = () => {
    const { t } = useTranslation();
    const message = useErrorMessage();

    return (
        <View height={'100vh'}>
            <IllustratedMessage>
                <NotFound />
                <Heading>{message}</Heading>

                <Button
                    variant={'accent'}
                    marginTop={'size-200'}
                    onPress={() => {
                        redirectTo(paths.root({}));
                    }}
                >
                    {t('application.errorPage.goHome')}
                </Button>
            </IllustratedMessage>
        </View>
    );
};
