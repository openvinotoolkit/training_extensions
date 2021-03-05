from action_recognition.video_reader import make_video_reader, ImageDirReader, opencv_read_image, pil_read_image, \
    accimage_read_image, VideoFileReader
import numpy as np


class TestMakeVideoReader:
    def test_returns_frame_reader_by_default(self):
        video_reader = make_video_reader()
        assert isinstance(video_reader, ImageDirReader)

    def test_returns_opencv_by_default(self):
        video_reader = make_video_reader()
        assert video_reader.read_image_fn is opencv_read_image


class TestImageDirReader:
    def test_reads_image(self, mocker):
        mocker.patch('os.path.exists')
        read_img_mock = mocker.Mock()
        img_dir_reader = ImageDirReader(read_img_mock, "/foo/image_%05d.jpg")

        ret = img_dir_reader.read('/foo/', [1, 2, 3])

        assert read_img_mock.call_count == 3
        assert len(ret) == 3
        read_img_mock.assert_has_calls([
            mocker.call('/foo/image_00001.jpg'),
            mocker.call('/foo/image_00002.jpg'),
            mocker.call('/foo/image_00003.jpg'),
        ])


class _MockVideoCapture:
    def __init__(self, *args, **kwargs):
        self._i = 0
        self._num_reads = 10

    def read(self):
        if self._i < self._num_reads:
            self._i += 1
            return True, self._i
        return False, None

    def release(self):
        pass


class TestVideoFileReader:
    def test_reads_frames(self, mocker):
        mocker.patch('cv2.VideoCapture', _MockVideoCapture)
        mocker.patch('cv2.cvtColor', lambda x, _: x)
        video_reader = VideoFileReader()

        frames = video_reader.read('/path', [1, 2, 3])

        assert [1, 2, 3] == frames

    def test_skips_frames(self, mocker):
        mocker.patch('cv2.VideoCapture', _MockVideoCapture)
        mocker.patch('cv2.cvtColor', lambda x, _: x)
        video_reader = VideoFileReader()

        frames = video_reader.read('/path', [2, 4, 6])

        assert [2, 4, 6] == frames


def test_opencv_reader_returns_rgb(data_path):
    img = opencv_read_image(str(data_path / 'rgb_test.png'))
    assert np.alltrue(img[:4, :4] == (255, 0, 0))
    assert np.alltrue(img[4:8, 4:8] == (0, 255, 0))
    assert np.alltrue(img[8:, 8:] == (0, 0, 255))


def test_pil_reader_returns_rgb(data_path):
    img = pil_read_image(str(data_path / 'rgb_test.png'))
    img = np.asarray(img)
    assert np.alltrue(img[:4, :4] == (255, 0, 0))
    assert np.alltrue(img[4:8, 4:8] == (0, 255, 0))
    assert np.alltrue(img[8:, 8:] == (0, 0, 255))
