// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { CSSProperties, Fragment, useMemo } from 'react';

import type { Label } from '@/api/types';
import { useTranslation } from '@/i18n';
import { Divider, Flex, Pressable, Text, Tooltip, TooltipTrigger } from '@geti-ui/ui';
import { clsx } from 'clsx';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { isEmpty } from 'lodash-es';
import { useHotkeys } from 'react-hotkeys-hook';

import { EMPTY_LABEL_ID } from '../../../shared/annotator/labels';
import { formatHotkeyForDisplay } from '../../../shared/hotkeys-definition';
import { usePinnedLabels } from './hooks/use-pinned-labels.hook';
import { LabelsEditorPopover } from './labels-editor/labels-editor-popover.component';
import { useLabels } from './use-labels.hook';

import classes from './labels.module.scss';

const MAX_VISIBLE_LABELS = 5;

type LabelBadgeProps = {
    label: Label;
    isSelected: boolean;
    onClick: () => void;
};

const LabelBadge = ({ label, isSelected, onClick }: LabelBadgeProps) => {
    const { t } = useTranslation();

    return (
        <TooltipTrigger isDisabled={isEmpty(label.hotkey)}>
            <Pressable>
                <button
                    onClick={onClick}
                    style={{ '--labelBgColor': label.color } as CSSProperties}
                    className={clsx(classes.badge, { [classes.selected]: isSelected })}
                    aria-pressed={isSelected}
                    aria-label={`Label ${label.name}`}
                >
                    <Text UNSAFE_className={classes.badgeText}>{label.name}</Text>
                </button>
            </Pressable>
            <Tooltip>
                {t('labels.toolbar.hotkeyTooltip', { hotkey: formatHotkeyForDisplay(label.hotkey ?? '') })}
            </Tooltip>
        </TooltipTrigger>
    );
};

type LabelsProps = {
    isClassification?: boolean;
    isMultiLabel?: boolean;
};

type LabelHotkeyBindingProps = {
    label: Label;
    onTrigger: (label: Label) => void;
};

const LabelHotkeyBinding = ({ label, onTrigger }: LabelHotkeyBindingProps) => {
    const hotkey = label.hotkey ?? '';

    useHotkeys(hotkey, () => onTrigger(label), [label, onTrigger]);

    return null;
};

export const Labels = ({ isClassification = false, isMultiLabel = false }: LabelsProps) => {
    const { t } = useTranslation();
    const { labels, hasLabels, toggleLabelOnAnnotations, isLabelActive, editableLabels } = useLabels({
        isClassification,
        isMultiLabel,
    });

    const projectId = useProjectIdentifier();
    const { isPinned, hasPinnedLabels } = usePinnedLabels(projectId);

    const visibleLabels = useMemo(() => {
        if (hasPinnedLabels) {
            return labels.filter((label) => isPinned(label.id) || label.id === EMPTY_LABEL_ID);
        }

        const emptyLabel = labels.find((label) => label.id === EMPTY_LABEL_ID);
        const visible = editableLabels.slice(0, MAX_VISIBLE_LABELS);

        return emptyLabel ? [...visible, emptyLabel] : visible;
    }, [labels, editableLabels, hasPinnedLabels, isPinned]);

    const hiddenLabelsCount = useMemo(() => {
        const visibleNonEmptyCount = visibleLabels.filter((label) => label.id !== EMPTY_LABEL_ID).length;
        return editableLabels.length - visibleNonEmptyCount;
    }, [editableLabels, visibleLabels]);

    return (
        <Flex alignItems='start' gap='size-100' minWidth={0} flex='1'>
            {labels
                .filter((label) => !isEmpty(label.hotkey))
                .map((label) => (
                    <LabelHotkeyBinding key={`hotkey-${label.id}`} label={label} onTrigger={toggleLabelOnAnnotations} />
                ))}

            {hasLabels && (
                <div aria-label={'Labels'} className={classes.labelsContainer}>
                    {visibleLabels.map((label) => (
                        <Fragment key={label.id}>
                            {label.id === EMPTY_LABEL_ID && <Divider size={'S'} orientation={'vertical'} />}
                            <LabelBadge
                                label={label}
                                isSelected={isLabelActive(label)}
                                onClick={() => toggleLabelOnAnnotations(label)}
                            />
                        </Fragment>
                    ))}
                    {hiddenLabelsCount > 0 && (
                        <Text UNSAFE_className={classes.overflowCount}>
                            {t('labels.toolbar.overflowCount', { count: hiddenLabelsCount })}
                        </Text>
                    )}
                </div>
            )}
            <LabelsEditorPopover
                isClassification={isClassification}
                isMultiLabel={isMultiLabel}
                hasLabels={hasLabels}
            />
        </Flex>
    );
};
