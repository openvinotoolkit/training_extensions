// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, Radio, RadioGroup, Text } from '@geti-ui/ui';
import { InfoOutline } from '@geti-ui/ui/icons';
import { useTranslation } from 'react-i18next';

import classes from './classification-task-type-selection.module.scss';

export type ClassificationTaskType = 'single-label' | 'multi-label';

type ClassificationTaskTypeItemProps = {
    type: ClassificationTaskType;
    label: string;
    description: string;
    onSelect: (type: ClassificationTaskType) => void;
    warning?: string;
};

const ClassificationTaskTypeItem = ({
    type,
    label,
    description,
    onSelect,
    warning,
}: ClassificationTaskTypeItemProps) => {
    return (
        <div onClick={() => onSelect(type)} className={classes.itemType}>
            <Radio value={type}>{label}</Radio>

            <Text UNSAFE_className={classes.typeDescription}>{description}</Text>
            {warning && (
                <Flex alignItems={'center'} gap={'size-50'} marginTop={'size-100'}>
                    <InfoOutline />
                    <Text UNSAFE_className={classes.warning}>{warning}</Text>
                </Flex>
            )}
        </div>
    );
};

type ClassificationTaskSelectionProps = {
    selectedType: ClassificationTaskType;
    onSelectedTypeChange: (type: ClassificationTaskType) => void;
};

export const ClassificationTaskSelection = ({
    selectedType,
    onSelectedTypeChange,
}: ClassificationTaskSelectionProps) => {
    const { t } = useTranslation();

    return (
        <Flex direction={'column'} gap={'size-250'}>
            <Text UNSAFE_className={classes.title}>{t('project.create.classificationType.title')}</Text>
            <RadioGroup
                isEmphasized
                aria-label={t('project.create.classificationType.groupLabel')}
                value={selectedType}
                onChange={(value) => onSelectedTypeChange(value as ClassificationTaskType)}
            >
                <Flex justifyContent={'center'}>
                    <Flex gap={'size-200'} width={'80%'}>
                        <ClassificationTaskTypeItem
                            type={'single-label'}
                            label={t('project.create.classificationType.singleLabel.label')}
                            description={t('project.create.classificationType.singleLabel.description')}
                            onSelect={onSelectedTypeChange}
                            warning={t('project.create.classificationType.singleLabel.warning')}
                        />
                        <ClassificationTaskTypeItem
                            type={'multi-label'}
                            label={t('project.create.classificationType.multiLabel.label')}
                            description={t('project.create.classificationType.multiLabel.description')}
                            onSelect={onSelectedTypeChange}
                        />
                    </Flex>
                </Flex>
            </RadioGroup>
        </Flex>
    );
};
