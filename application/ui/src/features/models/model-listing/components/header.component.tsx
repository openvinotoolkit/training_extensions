// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Key } from 'react';

import { ActionButton, Flex, Grid, Item, Menu, MenuTrigger, Picker } from '@geti-ui/ui';
import { MoreMenu } from '@geti-ui/ui/icons';

import { TrainModel } from '../../train-model/train-model.component';
import { useModelListing } from '../provider/model-listing-provider';
import type { GroupByMode } from '../types';
import { ExpandableSearch } from './expandable-search/expandable-search.component';

type MoreOptionsProps = {
    showFailedModels: boolean;
    onToggleShowFailedModels: () => void;
};
const MoreOptions = ({ showFailedModels, onToggleShowFailedModels }: MoreOptionsProps) => {
    const handleOptionsAction = (key: Key) => {
        switch (key) {
            case 'show-failed':
                onToggleShowFailedModels();
                break;
            default:
                break;
        }
    };

    return (
        <MenuTrigger>
            <ActionButton isQuiet aria-label={'Model listing options'}>
                <MoreMenu />
            </ActionButton>
            <Menu onAction={handleOptionsAction} aria-label={'Model listing options menu'}>
                <Item key={'show-failed'}>{showFailedModels ? 'Hide failed models' : 'Show failed models'}</Item>
            </Menu>
        </MenuTrigger>
    );
};

export const Header = () => {
    const { groupBy, onGroupByChange, searchBy, onSearchChange, showFailedModels, onToggleShowFailedModels } =
        useModelListing();

    return (
        <Grid columns={['auto auto 1fr']} gap={'size-100'} alignItems={'center'}>
            <Picker
                placeholder={'Group by'}
                width={'size-2400'}
                aria-label={'Group models'}
                selectedKey={groupBy}
                onSelectionChange={(key) => onGroupByChange(key as GroupByMode)}
            >
                <Item key='dataset'>Group by: Dataset</Item>
                <Item key='architecture'>Group by: Architecture</Item>
            </Picker>

            <MoreOptions showFailedModels={showFailedModels} onToggleShowFailedModels={onToggleShowFailedModels} />

            <Flex marginStart={'auto'} gap={'size-100'}>
                <ExpandableSearch value={searchBy} onChange={onSearchChange} />
                <TrainModel />
            </Flex>
        </Grid>
    );
};
