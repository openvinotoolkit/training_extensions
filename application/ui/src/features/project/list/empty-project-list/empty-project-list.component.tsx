// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Trans, useTranslation } from '@/i18n';
import { Button, Flex, Text } from '@geti-ui/ui';
import { useNavigate } from 'react-router-dom';

import { ReactComponent as EmptyFolderImage } from '../../../../assets/empty-folder.svg';
import { paths } from '../../../../constants/paths';
import { useImportDatasetDialog } from '../../providers/import-dataset-dialog-provider.component';
import { ImportDatasetAsNewProject } from '../import-dataset-as-new-project/import-dataset-as-new-project.component';
import { WorkflowSteps } from './workflow-steps.component';

import classes from './empty-project-list.module.scss';

export const EmptyProjectList = () => {
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
        <div className={classes.emptyState}>
            <p className={classes.intro}>
                <Trans
                    i18nKey='project.list.empty.intro'
                    components={{
                        name: <span className={classes.introName} />,
                        highlight: <span className={classes.introHighlight} />,
                    }}
                />
            </p>

            <Flex
                gap={'size-100'}
                direction={'column'}
                alignItems={'center'}
                justifyContent={'center'}
                UNSAFE_className={classes.container}
            >
                <EmptyFolderImage aria-label='empty list' />

                <Flex alignItems={'center'} gap={'size-100'}>
                    <Button variant='accent' id='create-new-project-button' onPress={handleCreateProject}>
                        <Text UNSAFE_style={{ whiteSpace: 'nowrap' }}>{t('project.list.empty.createNewProject')}</Text>
                    </Button>
                    <Button variant='accent' id='create-from-dataset-button' onPress={handleCreateFromDataset}>
                        <Text UNSAFE_style={{ whiteSpace: 'nowrap' }}>{t('project.list.empty.createFromDataset')}</Text>
                    </Button>
                </Flex>

                <ImportDatasetAsNewProject dialogState={datasetImportDialogState} />
            </Flex>

            <WorkflowSteps />
        </div>
    );
};
