// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { Divider, Flex, Switch, Text, View } from '@geti-ui/ui';

import { AnnotationSetting } from './annotation-setting.component';
import { CanvasSettingsState } from './canvas-settings-provider.component';
import { ImageSetting } from './image-setting.component';

interface SettingsListProps {
    canvasSettings: CanvasSettingsState;
    onCanvasSettingsChange: (canvasSettings: CanvasSettingsState) => void;
}

export const SettingsList = ({ canvasSettings, onCanvasSettingsChange }: SettingsListProps) => {
    const { t } = useTranslation();

    const updateCanvasSettings = <T extends keyof CanvasSettingsState>(
        key: T,
        value: CanvasSettingsState[T]['value']
    ) => {
        const newCanvasSettings = structuredClone(canvasSettings);
        newCanvasSettings[key].value = value;
        onCanvasSettingsChange(newCanvasSettings);
    };

    return (
        <View paddingEnd={'size-50'}>
            <Flex alignItems={'center'} justifyContent={'space-between'}>
                <Text>{t('annotator.canvas.hideLabels')}</Text>
                <Flex alignItems={'center'} gap={'size-100'}>
                    <Switch
                        aria-label={'Hide labels'}
                        isEmphasized
                        isSelected={canvasSettings.hideLabels.value}
                        onChange={(isSelected) => {
                            updateCanvasSettings('hideLabels', isSelected);
                        }}
                    />
                </Flex>
            </Flex>

            <Divider size={'S'} marginY={'size-250'} />

            <AnnotationSetting
                headerText={t('annotator.canvas.annotationFillOpacity')}
                ariaLabel={'Annotation fill opacity'}
                formatOptions={{ style: 'percent' }}
                defaultValue={canvasSettings.annotationFillOpacity.defaultValue}
                value={canvasSettings.annotationFillOpacity.value}
                handleValueChange={(value) => {
                    updateCanvasSettings('annotationFillOpacity', value);
                }}
            />
            <AnnotationSetting
                headerText={t('annotator.canvas.annotationBorderOpacity')}
                ariaLabel={'Annotation border opacity'}
                formatOptions={{ style: 'percent' }}
                defaultValue={canvasSettings.annotationBorderOpacity.defaultValue}
                value={canvasSettings.annotationBorderOpacity.value}
                handleValueChange={(value) => {
                    updateCanvasSettings('annotationBorderOpacity', value);
                }}
            />
            <Divider size={'S'} marginY={'size-250'} />
            <ImageSetting
                headerText={t('annotator.canvas.imageBrightness')}
                ariaLabel={'Image brightness'}
                formatOptions={{ signDisplay: 'exceptZero' }}
                defaultValue={canvasSettings.imageBrightness.defaultValue}
                value={canvasSettings.imageBrightness.value}
                handleValueChange={(value) => {
                    updateCanvasSettings('imageBrightness', value);
                }}
            />
            <ImageSetting
                headerText={t('annotator.canvas.imageContrast')}
                ariaLabel={'Image contrast'}
                formatOptions={{ signDisplay: 'exceptZero' }}
                defaultValue={canvasSettings.imageContrast.defaultValue}
                value={canvasSettings.imageContrast.value}
                handleValueChange={(value) => {
                    updateCanvasSettings('imageContrast', value);
                }}
            />
            <ImageSetting
                headerText={t('annotator.canvas.imageSaturation')}
                ariaLabel={'Image saturation'}
                formatOptions={{ signDisplay: 'exceptZero' }}
                defaultValue={canvasSettings.imageSaturation.defaultValue}
                value={canvasSettings.imageSaturation.value}
                handleValueChange={(value) => {
                    updateCanvasSettings('imageSaturation', value);
                }}
            />
            <Divider size={'S'} marginY={'size-250'} />

            <Flex alignItems={'center'} justifyContent={'space-between'}>
                <Text>{t('annotator.canvas.pixelView')}</Text>
                <Flex alignItems={'center'} gap={'size-100'}>
                    <Switch
                        aria-label={'Pixel view'}
                        isEmphasized
                        isSelected={canvasSettings.pixelView.value}
                        onChange={(isSelected) => {
                            updateCanvasSettings('pixelView', isSelected);
                        }}
                    />
                </Flex>
            </Flex>
        </View>
    );
};
