// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, screen, waitFor } from '@testing-library/react';
import { getMockedLabel } from 'mocks/mock-labels';
import { getMockedVideoFrame } from 'mocks/mock-media';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../../../../../api/utils';
import { server } from '../../../../../../../msw-node-setup';
import { EMPTY_LABEL_ID } from '../../../../../../../shared/annotator/labels';
import { VideoFrameSegment } from './video-frame-segment.component';

const FRAME_NUMBER = 0;
const FPS = 60;
const FRAME_COUNT = 600; // 10 seconds at 60 fps
const MODEL_ID = 'model-1';

const mockVideoFrame = getMockedVideoFrame({
    id: 'video-1',
    fps: FPS,
    frame_count: FRAME_COUNT,
    frame_number: FRAME_NUMBER,
    frame_stride: 1,
});

const labelA = getMockedLabel({ id: 'label-a', name: 'Cat', color: '#ff0000' });
const labelB = getMockedLabel({ id: 'label-b', name: 'Dog', color: '#00ff00' });
const labelC = getMockedLabel({ id: EMPTY_LABEL_ID, name: 'No object', color: '#FFF' });
const labels = [labelA, labelB, labelC];

vi.mock('../../../../video-player-provider.component', () => ({
    useVideoPlayer: () => ({
        videoFrame: mockVideoFrame,
        videoRef: { current: null },
        isMuted: false,
        toggleMute: vi.fn(),
        playbackRate: 1,
        changePlaybackRate: vi.fn(),
        videoControls: { play: vi.fn(), pause: vi.fn() },
        changeCurrentFrameIndex: vi.fn(),
        step: 1,
        changeStep: vi.fn(),
    }),
}));

vi.mock('../../../../../../../features/annotator/predictions-setup-provider.component', () => ({
    usePredictionSetup: () => ({
        selectedDevice: 'cpu',
        changeSelectedDevice: vi.fn(),

        selectedModelId: MODEL_ID,
        selectedModel: { id: MODEL_ID, name: 'Model 1 [FP32]', modelId: 'model-base-1' },
        changeSelectedModelId: vi.fn(),
        selectableModels: [{ id: MODEL_ID, name: 'Model 1 [FP32]', modelId: 'model-base-1' }],
    }),
}));

const renderSegment = (overrides: Partial<Parameters<typeof VideoFrameSegment>[0]> = {}) => {
    return render(
        <VideoFrameSegment
            mode='annotation'
            frameNumber={FRAME_NUMBER}
            labels={labels}
            colIndex={0}
            isFirstFrame={true}
            isLastFrame={false}
            isSelectedFrame={false}
            showTicks={true}
            onClick={vi.fn()}
            {...overrides}
        />
    );
};

describe('VideoFrameSegment', () => {
    describe('shared behaviour (mode-independent)', () => {
        describe('selected frame overlay', () => {
            it('renders the overlay when isSelectedFrame is true', () => {
                renderSegment({ isSelectedFrame: true });

                expect(screen.getByTestId('selected')).toBeInTheDocument();
            });

            it('does not render the overlay when isSelectedFrame is false', () => {
                renderSegment({ isSelectedFrame: false });

                expect(screen.queryByTestId('selected')).not.toBeInTheDocument();
            });
        });

        describe('tick display', () => {
            it('shows the frame number tick when showTicks is true', () => {
                renderSegment({ showTicks: true, frameNumber: 42 });

                expect(screen.getByText('42f')).toBeInTheDocument();
            });

            it('does not show the tick when showTicks is false', () => {
                renderSegment({ showTicks: false, frameNumber: 42 });

                expect(screen.queryByText('42f')).not.toBeInTheDocument();
            });
        });

        describe('click interaction', () => {
            it('calls onClick with the frame number when the segment is clicked', () => {
                const mockOnClick = vi.fn();
                renderSegment({ onClick: mockOnClick, frameNumber: 120, showTicks: true });

                fireEvent.click(screen.getByText('120f'));

                expect(mockOnClick).toHaveBeenCalledWith(120);
            });
        });
    });

    describe('annotation mode', () => {
        describe('when the frame has no annotations', () => {
            beforeEach(() => {
                server.use(
                    http.get('/api/projects/{project_id}/dataset/media/{media_id}/frames', () => {
                        return HttpResponse.json([
                            {
                                media_id: 'video-1',
                                frame_index: FRAME_NUMBER,
                                annotation_data: {
                                    annotations: [],
                                    user_reviewed: false,
                                    prediction_model_id: null,
                                    media_id: 'video-1',
                                    subset: 'training',
                                },
                            },
                        ]);
                    })
                );
            });

            it('renders a label row for each label', async () => {
                renderSegment({ mode: 'annotation' });

                await waitFor(() => {
                    expect(
                        screen.getByRole('gridcell', { name: `Label ${labelA.name} in frame number ${FRAME_NUMBER}` })
                    ).toBeInTheDocument();
                    expect(
                        screen.getByRole('gridcell', { name: `Label ${labelB.name} in frame number ${FRAME_NUMBER}` })
                    ).toBeInTheDocument();
                    expect(
                        screen.getByRole('gridcell', { name: `Label ${labelC.name} in frame number ${FRAME_NUMBER}` })
                    ).toBeInTheDocument();
                });
            });

            it('marks each label segment as "Not labeled label" when no annotations are present', async () => {
                renderSegment({ mode: 'annotation' });

                await waitFor(() => {
                    expect(screen.getAllByRole('presentation', { name: 'Not labeled' })).toHaveLength(
                        [labelA, labelB].length
                    );
                });
            });

            it('marks segment that have the empty label as "No object"', async () => {
                renderSegment({ mode: 'annotation' });

                await waitFor(() => {
                    expect(screen.getByRole('presentation', { name: labelC.name })).toBeInTheDocument();
                });
            });
        });

        describe('when the frame has annotations for some labels', () => {
            beforeEach(() => {
                server.use(
                    http.get('/api/projects/{project_id}/dataset/media/{media_id}/frames', () => {
                        return HttpResponse.json([
                            {
                                media_id: 'video-1',
                                frame_index: FRAME_NUMBER,
                                annotation_data: {
                                    annotations: [
                                        {
                                            shape: { type: 'full_image' },
                                            labels: [{ id: labelA.id }],
                                            confidences: null,
                                        },
                                    ],
                                    user_reviewed: true,
                                    prediction_model_id: null,
                                    media_id: 'video-1',
                                    subset: 'training',
                                },
                            },
                        ]);
                    })
                );
            });

            it('shows the annotated label name for the matched label', async () => {
                renderSegment({ mode: 'annotation' });

                await waitFor(() => {
                    expect(screen.getByRole('presentation', { name: labelA.name })).toBeInTheDocument();
                });
            });

            it('shows "Not labeled" for the unmatched label', async () => {
                renderSegment({ mode: 'annotation' });

                await waitFor(() => {
                    expect(screen.getAllByRole('presentation', { name: 'Not labeled' })).toHaveLength(
                        [labelB, labels].length
                    );
                });
            });
        });
    });

    describe('prediction mode', () => {
        describe('when the frame has no predictions', () => {
            beforeEach(() => {
                server.use(
                    http.post('/api/projects/{project_id}/dataset/media:predict', () => {
                        return HttpResponse.json({
                            predictions: [
                                {
                                    media: { id: 'video-1', frame_index: FRAME_NUMBER },
                                    prediction: [],
                                },
                            ],
                        });
                    })
                );
            });

            it('renders a label row for each label', async () => {
                renderSegment({ mode: 'prediction' });

                await waitFor(() => {
                    expect(
                        screen.getByRole('gridcell', { name: `Label ${labelA.name} in frame number ${FRAME_NUMBER}` })
                    ).toBeInTheDocument();
                    expect(
                        screen.getByRole('gridcell', { name: `Label ${labelB.name} in frame number ${FRAME_NUMBER}` })
                    ).toBeInTheDocument();
                    expect(
                        screen.getByRole('gridcell', { name: `Label ${labelC.name} in frame number ${FRAME_NUMBER}` })
                    ).toBeInTheDocument();
                });
            });

            it('marks each label segment as "No prediction" when no predictions exist', async () => {
                renderSegment({ mode: 'prediction' });

                await waitFor(() => {
                    expect(screen.getAllByRole('presentation', { name: 'No prediction' })).toHaveLength(
                        [labelA, labelB].length
                    );
                });
            });

            it('marks segment that have the empty label as "Predicted No object"', async () => {
                renderSegment({ mode: 'prediction' });

                await waitFor(() => {
                    expect(screen.getByRole('presentation', { name: `Predicted ${labelC.name}` })).toBeInTheDocument();
                });
            });
        });

        describe('when the frame has predictions for some labels', () => {
            beforeEach(() => {
                server.use(
                    http.post('/api/projects/{project_id}/dataset/media:predict', () => {
                        return HttpResponse.json({
                            predictions: [
                                {
                                    media: { id: 'video-1', frame_index: FRAME_NUMBER },
                                    prediction: [
                                        {
                                            shape: { type: 'full_image' },
                                            labels: [{ id: labelB.id }],
                                            confidences: [0.95],
                                        },
                                    ],
                                },
                            ],
                        });
                    })
                );
            });

            it('shows the predicted label name for the matched label', async () => {
                renderSegment({ mode: 'prediction' });

                await waitFor(() => {
                    expect(screen.getByRole('presentation', { name: `Predicted ${labelB.name}` })).toBeInTheDocument();
                });
            });

            it('shows "No prediction" for the unmatched label', async () => {
                renderSegment({ mode: 'prediction' });

                await waitFor(() => {
                    expect(screen.getAllByRole('presentation', { name: 'No prediction' })).toHaveLength(
                        [labelA, labelC].length
                    );
                });
            });
        });
    });
});
