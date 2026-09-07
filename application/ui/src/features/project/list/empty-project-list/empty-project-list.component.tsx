// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, Flex, Text } from '@geti-ui/ui';
import { useNavigate } from 'react-router-dom';

import { ReactComponent as EmptyFolderImage } from '../../../../assets/empty-folder.svg';
import { paths } from '../../../../constants/paths';
import { useImportDatasetDialog } from '../../providers/import-dataset-dialog-provider.component';
import { ImportDatasetAsNewProject } from '../import-dataset-as-new-project/import-dataset-as-new-project.component';
import { WorkflowSteps } from './workflow-steps.component';

import classes from './empty-project-list.module.scss';

export const EmptyProjectList = () => {
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
                Welcome to <span className={classes.introName}>Geti</span>, a vision AI platform that guides you through
                the <span className={classes.introHighlight}>complete model lifecycle</span>, from dataset preparation
                to training, optimization and deployment.
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
                        <Text UNSAFE_style={{ whiteSpace: 'nowrap' }}>Create new Project</Text>
                    </Button>
                    <Button variant='accent' id='create-from-dataset-button' onPress={handleCreateFromDataset}>
                        <Text UNSAFE_style={{ whiteSpace: 'nowrap' }}>Create from dataset</Text>
                    </Button>
                </Flex>

                <ImportDatasetAsNewProject dialogState={datasetImportDialogState} />
            </Flex>

            <WorkflowSteps />
        </div>
    );
};
