// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { IconWrapper } from '@/components/icon-wrapper/icon-wrapper.component';
import { ActionButton, Flex, Heading, IllustratedMessage, Text, Tooltip, TooltipTrigger, View } from '@geti-ui/ui';
import { useHotkeys } from 'react-hotkeys-hook';

import type { ToolConfig, ToolType } from '../interface';

import classes from './tool.module.scss';

type HotkeyProps = {
    hotkey: string;
};

const Hotkey = ({ hotkey }: HotkeyProps) => {
    return (
        <View backgroundColor={'gray-200'} paddingX={'size-100'} paddingY={'size-50'}>
            <Text UNSAFE_className={classes.hotkeyText}>{hotkey.toLocaleUpperCase()}</Text>
        </View>
    );
};

type DrawingToolsTooltipProps = {
    tool: ToolConfig;
};

const DrawingToolsTooltip = ({ tool }: DrawingToolsTooltipProps) => {
    const { tooltip, hotkey, label } = tool;

    return (
        <IllustratedMessage>
            <img className={classes.drawingToolsTooltipsImg} src={tooltip?.img} alt={label} />
            <View UNSAFE_className={classes.drawingToolsTooltipsContent}>
                <Flex alignItems={'center'} justifyContent={'space-between'} order={2}>
                    <Heading UNSAFE_className={classes.drawingToolsTooltipsTitle}>{label}</Heading>
                    {hotkey !== undefined && <Hotkey hotkey={hotkey} />}
                </Flex>
                <Text UNSAFE_className={classes.drawingToolsTooltipsDescription}>{tooltip?.description}</Text>
            </View>
        </IllustratedMessage>
    );
};

type ToolProps = {
    tool: ToolConfig;
    activeTool: ToolType | null;
    setActiveTool: (tool: ToolType) => void;
    isDisabled?: boolean;
};

export const Tool = ({ tool, activeTool, setActiveTool, isDisabled }: ToolProps) => {
    useHotkeys(tool.hotkey, () => setActiveTool(tool.type), [setActiveTool, isDisabled], { enabled: !isDisabled });

    const label = `${tool.label} (${tool.hotkey})`;

    return (
        <TooltipTrigger placement={'right'}>
            <ActionButton
                isQuiet
                width={'size-400'}
                onPress={() => setActiveTool(tool.type)}
                aria-label={label}
                isDisabled={isDisabled}
                aria-pressed={activeTool === tool.type}
            >
                <IconWrapper isSelected={activeTool === tool.type} isDisabled={isDisabled}>
                    <tool.icon data-tool={tool.type} />
                </IconWrapper>
            </ActionButton>
            <Tooltip UNSAFE_className={tool.tooltip ? classes.drawingToolsTooltips : undefined}>
                {tool.tooltip === undefined ? label : <DrawingToolsTooltip tool={tool} />}
            </Tooltip>
        </TooltipTrigger>
    );
};
