// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { MediaWithPagination, Pagination } from '@/api/types';
import { InfiniteData, QueryClient } from '@tanstack/react-query';
import { getMockedMediaImage, getMockedVideo } from 'mocks/mock-media';

import { incrementCachedAnnotatedFrameCount } from './increment-cached-annotated-frame-count';

const createQueryClient = () => new QueryClient();

const getMockedPagination = (overrides: Partial<Pagination>) => ({
    total: 1,
    offset: 0,
    limit: 0,
    count: 0,
    ...overrides,
});

const MEDIA_QUERY_KEY = ['get', '/api/projects/{project_id}/dataset/media'] as const;

const setMediaQueryData = (queryClient: QueryClient, pages: MediaWithPagination[]) => {
    queryClient.setQueryData<InfiniteData<MediaWithPagination>>(MEDIA_QUERY_KEY, {
        pages,
        pageParams: pages.map((_, i) => i),
    });
};

const getMediaQueryData = (queryClient: QueryClient) => {
    return queryClient.getQueryData<InfiniteData<MediaWithPagination>>(MEDIA_QUERY_KEY);
};

describe('incrementCachedAnnotatedFrameCount', () => {
    it('increments annotated_frame_count for the matching video', () => {
        const queryClient = createQueryClient();
        const video = getMockedVideo({ id: 'video-1', annotated_frame_count: 5 });

        setMediaQueryData(queryClient, [
            {
                items: [video],
                pagination: getMockedPagination({ total: 1 }),
            },
        ]);

        incrementCachedAnnotatedFrameCount(queryClient, video);

        const data = getMediaQueryData(queryClient);
        expect(data?.pages[0].items[0]).toEqual(
            expect.objectContaining({ annotated_frame_count: video.annotated_frame_count + 1 })
        );
    });

    it('does not modify non-matching videos', () => {
        const queryClient = createQueryClient();
        const targetVideo = getMockedVideo({ id: 'video-1', annotated_frame_count: 5 });
        const otherVideo = getMockedVideo({ id: 'video-2', annotated_frame_count: 10 });

        setMediaQueryData(queryClient, [
            { items: [targetVideo, otherVideo], pagination: getMockedPagination({ total: 2 }) },
        ]);

        incrementCachedAnnotatedFrameCount(queryClient, targetVideo);

        const data = getMediaQueryData(queryClient);
        expect(data?.pages[0].items).toEqual([
            expect.objectContaining({ annotated_frame_count: targetVideo.annotated_frame_count + 1 }),
            expect.objectContaining({ annotated_frame_count: otherVideo.annotated_frame_count }),
        ]);
    });

    it('does not modify image items', () => {
        const queryClient = createQueryClient();
        const image = getMockedMediaImage({ id: 'image-1' });
        const video = getMockedVideo({ id: 'video-1', annotated_frame_count: 3 });

        setMediaQueryData(queryClient, [{ items: [image, video], pagination: getMockedPagination({ total: 2 }) }]);

        incrementCachedAnnotatedFrameCount(queryClient, { ...image, type: 'image' });

        const data = getMediaQueryData(queryClient);
        expect(data?.pages[0].items[0]).toEqual(expect.objectContaining({ id: image.id }));
    });

    it('handles multiple pages', () => {
        const queryClient = createQueryClient();
        const video1 = getMockedVideo({ id: 'video-1', annotated_frame_count: 2 });
        const video2 = getMockedVideo({ id: 'video-2', annotated_frame_count: 7 });

        setMediaQueryData(queryClient, [
            { items: [video1], pagination: getMockedPagination({ total: 2 }) },
            { items: [video2], pagination: getMockedPagination({ total: 2 }) },
        ]);

        incrementCachedAnnotatedFrameCount(queryClient, video2);

        const data = getMediaQueryData(queryClient);

        expect(data?.pages[0].items[0]).toEqual(
            expect.objectContaining({ annotated_frame_count: video1.annotated_frame_count })
        );
        expect(data?.pages[1].items[0]).toEqual(
            expect.objectContaining({ annotated_frame_count: video2.annotated_frame_count + 1 })
        );
    });
});
