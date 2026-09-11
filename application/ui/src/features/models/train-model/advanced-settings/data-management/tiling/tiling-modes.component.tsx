// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { Content, ContextualHelp, Text, ToggleButtons } from '@geti-ui/ui';

import { getTilingModeLabel, TILING_MODES, TilingMode } from './utils';

import classes from './tiling.module.scss';

type TilingModeContextualHelpProps = {
    description: string;
};

const TilingModeContextualHelp = ({ description }: TilingModeContextualHelpProps) => {
    return (
        <ContextualHelp variant='info'>
            <Content>
                <Text>{description}</Text>
            </Content>
        </ContextualHelp>
    );
};

type TilingModesProps = {
    description: string;
    selectedTilingMode: TilingMode;
    onTilingModeChange: (newTilingMode: TilingMode) => void;
};

export const TilingModes = ({ description, selectedTilingMode, onTilingModeChange }: TilingModesProps) => {
    const { t } = useTranslation();

    return (
        <>
            <Text UNSAFE_className={classes.title}>
                {t('models.training.dataManagement.tiling.modeLabel')}{' '}
                <TilingModeContextualHelp description={description} />
            </Text>
            <ToggleButtons
                options={[TILING_MODES.OFF, TILING_MODES.AUTOMATIC, TILING_MODES.CUSTOM]}
                selectedOption={selectedTilingMode}
                onOptionChange={onTilingModeChange}
                getLabel={(option) => getTilingModeLabel(option, t)}
            />
        </>
    );
};
