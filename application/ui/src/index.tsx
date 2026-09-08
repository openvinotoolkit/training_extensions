// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import React from 'react';

import ReactDOM from 'react-dom/client';

import './i18n';

import { setupStorageCleanup } from './platform/storage-cleanup';
import { Providers } from './providers';

import './index.css';
import './shared/styles/view-transitions.scss';

setupStorageCleanup();

const rootEl = document.getElementById('root');
if (rootEl) {
    const root = ReactDOM.createRoot(rootEl);
    root.render(
        <React.StrictMode>
            <Providers />
        </React.StrictMode>
    );
}
