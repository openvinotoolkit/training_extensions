from action_recognition.dataset import sample_clips


def _make_video(length):
    return {
        'segment': (1, length),
        'frame_indices': list(range(1, 1 + length)),
        'video': '1'
    }


def _check_video_segment(video, start, expected_len):
    assert video['frame_indices'] == list(range(start, start + expected_len))


class TestSampleClips:
    def test_step_one(self):
        videos = [_make_video(5)]

        clips = sample_clips(videos, 3, 3)

        assert len(clips) == 3
        _check_video_segment(clips[0], 1, 3)
        _check_video_segment(clips[1], 2, 3)
        _check_video_segment(clips[2], 3, 3)

    def test_step_two(self):
        videos = [_make_video(7)]

        clips = sample_clips(videos, 3, 3)

        assert len(clips) == 3
        _check_video_segment(clips[0], 1, 3)
        _check_video_segment(clips[1], 3, 3)
        _check_video_segment(clips[2], 5, 3)

    def test_one_sample(self):
        videos = [_make_video(16)]

        clips = sample_clips(videos, 1, 1)

        assert len(clips) == 1
        assert clips[0]['frame_indices'] == videos[0]['frame_indices']

    def test_short_clip(self):
        videos = [_make_video(3)]

        clips = sample_clips(videos, 4, 3)

        assert len(clips) == 3
        _check_video_segment(clips[0], 1, 3)
        _check_video_segment(clips[1], 2, 2)
        _check_video_segment(clips[2], 3, 1)
