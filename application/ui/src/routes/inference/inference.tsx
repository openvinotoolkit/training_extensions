// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ZoomProvider } from '@/components/zoom/zoom.provider';
import { Grid } from '@geti-ui/ui';

import { Sidebar } from '../../features/inference/aside/sidebar-tabs.component';
import { Footer } from '../../features/inference/footer/footer.component';
import { Header } from '../../features/inference/header/inference-header.component';
import { StreamContainer } from '../../features/inference/stream/stream-container';

export const Inference = () => {
    return (
        <Grid
            areas={['toolbar aside', 'canvas aside', 'footer aside']}
            rows={['size-800', 'minmax(0, 1fr)', 'size-600']}
            columns={['minmax(0, 1fr)', 'auto']}
            height={'100%'}
            gap={'size-10'}
            UNSAFE_style={{
                overflow: 'hidden',
            }}
        >
            <Header />
            <ZoomProvider>
                <StreamContainer />
            </ZoomProvider>
            <Sidebar />
            <Footer />
        </Grid>
    );
};
