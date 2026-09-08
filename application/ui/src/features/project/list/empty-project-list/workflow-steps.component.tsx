// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Text, View } from '@geti-ui/ui';
import { Adjustments, AICPUIcon, AutoTraining, Edit, FolderLight } from '@geti-ui/ui/icons';

import classes from './workflow-steps.module.scss';

const STEPS = [
    { label: 'Add data', Icon: FolderLight },
    { label: 'Annotate', Icon: Edit },
    { label: 'Train', Icon: AutoTraining },
    { label: 'Optimize', Icon: Adjustments },
    { label: 'Run inference', Icon: AICPUIcon },
];

export const WorkflowSteps = () => {
    return (
        <View UNSAFE_className={classes.workflow}>
            <ol aria-label='Geti workflow' className={classes.steps}>
                {STEPS.map(({ label, Icon }) => (
                    <li key={label} className={classes.step}>
                        <Text UNSAFE_className={classes.circle}>
                            <Icon aria-hidden />
                        </Text>
                        {label}
                    </li>
                ))}
            </ol>

            <View UNSAFE_className={classes.loop}>
                <Text UNSAFE_className={classes.loopText}>
                    Monitor predictions and collect more data to iteratively fine-tune your model
                </Text>
            </View>
        </View>
    );
};
