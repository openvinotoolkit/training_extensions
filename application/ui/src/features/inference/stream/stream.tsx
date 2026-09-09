// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useRef, useState, type SyntheticEvent } from 'react';

import { ZoomTransform } from '@/components/zoom/zoom-transform';

import { useWebRTCConnection } from './web-rtc-connection-provider';

const useStreamToVideo = () => {
    const videoRef = useRef<HTMLVideoElement>(null);

    const { status, webRTCConnectionRef } = useWebRTCConnection();

    const connect = useCallback(async () => {
        const videoOutput = videoRef.current;
        const webrtcConnection = webRTCConnectionRef.current;
        const peerConnection = webrtcConnection?.getPeerConnection();

        if (!peerConnection) {
            return;
        }

        const receivers = peerConnection.getReceivers() ?? [];
        const stream = new MediaStream(receivers.map((receiver) => receiver.track).filter(Boolean));

        if (videoOutput && videoOutput.srcObject !== stream) {
            videoOutput.srcObject = stream;
        }
    }, [webRTCConnectionRef]);

    useEffect(() => {
        if (status !== 'connected') return;

        const peerConnection = webRTCConnectionRef.current?.getPeerConnection();
        if (!peerConnection) return;

        connect();
        peerConnection.addEventListener('track', connect);

        return () => {
            peerConnection.removeEventListener('track', connect);
        };
    }, [status, webRTCConnectionRef, connect]);

    return videoRef;
};

const DEFAULT_VIDEO_SIZE = { width: 1280, height: 720 };

export const Stream = () => {
    const videoRef = useStreamToVideo();
    const [videoSize, setVideoSize] = useState(DEFAULT_VIDEO_SIZE);

    const onLoadedMetadata = (e: SyntheticEvent<HTMLVideoElement>) => {
        const { videoWidth, videoHeight } = e.currentTarget;

        if (videoWidth && videoHeight) {
            setVideoSize({ width: videoWidth, height: videoHeight });
        }
    };

    const updateSizeFromVideo = (e: SyntheticEvent<HTMLVideoElement>) => {
        const video = e.currentTarget;
        const { videoWidth, videoHeight } = video;

        if (videoWidth && videoHeight) {
            setVideoSize((prev) =>
                prev.width === videoWidth && prev.height === videoHeight
                    ? prev
                    : { width: videoWidth, height: videoHeight }
            );
        }
    };

    return (
        <ZoomTransform target={videoSize}>
            {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
            <video
                ref={videoRef}
                autoPlay
                playsInline
                controls={false}
                width={videoSize.width}
                height={videoSize.height}
                onLoadedMetadata={onLoadedMetadata}
                onResize={updateSizeFromVideo}
            />
        </ZoomTransform>
    );
};
