// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { Item, Picker } from '@geti-ui/ui';

import { createTrainingDeviceKey, useTrainModelState } from '../train-model-provider.component';
import { createDeviceName } from './utils';

export const SelectTrainingDevice = () => {
    const { t } = useTranslation();
    const { trainingDevices, onSelectTrainingDevice, selectedTrainingDevice } = useTrainModelState();

    return (
        <Picker
            flex={1}
            items={trainingDevices}
            label={t('models.training.devices.selectLabel')}
            selectedKey={selectedTrainingDevice}
            onSelectionChange={(key) => key !== null && onSelectTrainingDevice(key.toString())}
        >
            {(item) => <Item key={createTrainingDeviceKey(item)}>{createDeviceName(item)}</Item>}
        </Picker>
    );
};
