// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Text, View } from '@geti-ui/ui';
import { Adjustments, AICPUIcon, AutoTraining, Edit, FolderLight } from '@geti-ui/ui/icons';
import { useTranslation } from 'react-i18next';

import classes from './workflow-steps.module.scss';

const STEPS = [
    { labelKey: 'project.list.workflow.addData', Icon: FolderLight },
    { labelKey: 'project.list.workflow.annotate', Icon: Edit },
    { labelKey: 'project.list.workflow.train', Icon: AutoTraining },
    { labelKey: 'project.list.workflow.optimize', Icon: Adjustments },
    { labelKey: 'project.list.workflow.runInference', Icon: AICPUIcon },
] as const;

export const WorkflowSteps = () => {
    const { t } = useTranslation();

    return (
        <View UNSAFE_className={classes.workflow}>
            <ol aria-label={t('project.list.workflow.label')} className={classes.steps}>
                {STEPS.map(({ labelKey, Icon }) => (
                    <li key={labelKey} className={classes.step}>
                        <Text UNSAFE_className={classes.circle}>
                            <Icon aria-hidden />
                        </Text>
                        {t(labelKey)}
                    </li>
                ))}
            </ol>

            <View UNSAFE_className={classes.loop}>
                <Text UNSAFE_className={classes.loopText}>{t('project.list.workflow.loop')}</Text>
            </View>
        </View>
    );
};
