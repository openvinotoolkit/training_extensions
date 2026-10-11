// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ExportDatasetConfig } from '@/components/export-dataset-config-dialog/export-dataset-config.component';
import { useTranslation } from '@/i18n';
import { Button, Item, Key, Menu, MenuTrigger } from '@geti-ui/ui';
import { useOverlayTriggerState } from '@react-stately/overlays';
import { useOptionalDatasetViewsQuery } from 'hooks/api/dataset-views.hook';
import { ENTIRE_DATASET_VIEW_ID, useDatasetViewId } from 'hooks/use-dataset-view-id.hook';

import { useImportDatasetDialogState } from '../providers/export-import-dataset-dialog-provider.component';
import { MainDatasetStatistics } from './export-dataset/dataset-statistics.component';
import { ImportDatasetToProject } from './import-dataset/Import-dataset-to-project.component';

export const ImportExport = () => {
    const { t } = useTranslation();
    const exportDialogState = useOverlayTriggerState({});
    const { datasetImportDialogState, setCurrentStep } = useImportDatasetDialogState();

    const [datasetViewId] = useDatasetViewId();
    const isDatasetView = datasetViewId !== ENTIRE_DATASET_VIEW_ID;
    const { data: datasetViews } = useOptionalDatasetViewsQuery(isDatasetView);
    const datasetViewName = isDatasetView
        ? (datasetViews?.find(({ id }) => id === datasetViewId)?.name ?? t('dataset.views.deleted'))
        : undefined;

    const handleMenuAction = (option: Key) => {
        switch (option) {
            case 'export':
                exportDialogState.open();
                break;

            case 'import':
                setCurrentStep('uploading');
                datasetImportDialogState.open();
                break;
        }
    };

    return (
        <>
            <MenuTrigger>
                <Button variant='secondary' aria-label='import export dataset'>
                    {t('dataset.importExport.trigger')}
                </Button>
                <Menu onAction={handleMenuAction}>
                    <Item key='export'>
                        {isDatasetView ? t('dataset.export.headingDatasetView') : t('dataset.importExport.exportItem')}
                    </Item>
                    <Item key='import'>{t('dataset.import.dialogHeading')}</Item>
                </Menu>
            </MenuTrigger>

            <ImportDatasetToProject />

            <ExportDatasetConfig
                datasetId={null}
                datasetViewId={datasetViewId}
                datasetViewName={datasetViewName}
                dialogState={exportDialogState}
                statistics={<MainDatasetStatistics />}
            />
        </>
    );
};
