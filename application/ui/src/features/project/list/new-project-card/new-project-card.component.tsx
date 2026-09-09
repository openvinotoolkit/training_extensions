// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { ActionButton, Flex, Text } from '@geti-ui/ui';
import { AddCircle } from '@geti-ui/ui/icons';
import { useNavigate } from 'react-router-dom';

import { paths } from '../../../../constants/paths';
import { useImportDatasetDialog } from '../../providers/import-dataset-dialog-provider.component';
import { ImportDatasetAsNewProject } from '../import-dataset-as-new-project/import-dataset-as-new-project.component';

import classes from './new-project-menu.module.scss';

export const NewProjectCard = () => {
    const { t } = useTranslation();
    const navigate = useNavigate();
    const { datasetImportDialogState, setCurrentStep, setCurrentStagedId } = useImportDatasetDialog();

    const handleCreateProject = () => {
        navigate(paths.project.new.pattern, {
            viewTransition: true,
        });
    };

    const handleCreateFromDataset = () => {
        setCurrentStep('uploading');
        setCurrentStagedId(null);
        datasetImportDialogState.open();
    };

    return (
        <Flex gap={'size-300'} height={'100%'}>
            <ActionButton onPress={handleCreateProject} UNSAFE_className={classes.button}>
                <AddCircle />
                <Text>{t('project.list.createNewProject')}</Text>
            </ActionButton>
            <ActionButton onPress={handleCreateFromDataset} UNSAFE_className={classes.button}>
                <AddCircle />
                <Text>{t('project.list.createProjectFromDataset')}</Text>
            </ActionButton>
            <ImportDatasetAsNewProject dialogState={datasetImportDialogState} />
        </Flex>
    );
};
