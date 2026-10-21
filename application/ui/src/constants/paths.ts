// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { generatePath, type PathParam } from 'react-router';

type RoutePath<Pattern extends string> = {
    (params: { [Param in PathParam<Pattern>]: string }): string;
    pattern: Pattern;
};

const route = <Pattern extends string>(pattern: Pattern): RoutePath<Pattern> =>
    Object.assign((params: { [Param in PathParam<Pattern>]: string }) => generatePath<string>(pattern, params), {
        pattern,
    });

export const paths = {
    root: route('/'),
    project: {
        index: route('/projects'),
        new: route('/projects/new'),
        details: route('/projects/:projectId'),
        inference: route('/projects/:projectId/inference'),
        dataset: {
            index: route('/projects/:projectId/dataset'),
            item: {
                index: route('/projects/:projectId/dataset/:datasetItemId'),
                frame: route('/projects/:projectId/dataset/:datasetItemId/:frameNumber'),
            },
        },
        models: route('/projects/:projectId/models'),
    },
};
