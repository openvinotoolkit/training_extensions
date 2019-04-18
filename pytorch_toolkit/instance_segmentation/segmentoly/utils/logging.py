"""
 Copyright (c) 2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import logging
import sys
from datetime import timedelta

from tensorboardX import SummaryWriter


def setup_logging(level=logging.INFO, file_path=None, stream=sys.stdout):
    log_format = '{levelname} {asctime} {filename}:{lineno:>4d}] {message}'
    date_format = '%d-%m-%y %H:%M:%S'
    handlers = [logging.StreamHandler(stream), ]
    if file_path:
        handlers.append(logging.FileHandler(file_path))
    logging.basicConfig(level=level, format=log_format, datefmt=date_format, style='{', handlers=handlers)


class TrainingLogger(object):
    def close(self):
        raise NotImplementedError

    def __call__(self, **kwargs):
        raise NotImplementedError


class TextLogger(TrainingLogger):
    def __init__(self, logger, *args, delimiter=', ', max_width=140, **kwargs):
        self.logger = logger
        self.delimiter = delimiter
        self.max_width = max_width

    def close(self):
        pass

    def _format_float(self, v):
        try:
            v = float(v)
            res = '{:.6}'.format(v)
        except (ValueError, TypeError):
            res = '{}'.format(v)
        return res

    def _format_value(self, k, v):
        return '{}: {}'.format(k, self._format_float(v))

    def _soft_wrap(self, line):
        lines = []
        while len(line) > self.max_width:
            p = line.rfind(self.delimiter, 0, self.max_width)
            if p > 0:
                lines.append(line[:p + len(self.delimiter)])
                line = line[p + len(self.delimiter):]
        return lines

    def _estimate_time(self, timer, step, total_steps):
        elapsed_time = timedelta(seconds=timer.total_time)
        elapsed_time = str(elapsed_time).split('.')[0]
        left_time = timedelta(seconds=timer.average_time * (total_steps - step))
        left_time = str(left_time).split('.')[0]
        secs_per_iter = timer.average_time
        return 'time elapsed/~left: {} / {} ({:.3} sec/it)'.format(elapsed_time, left_time, secs_per_iter)

    def __call__(self, step, total_steps, lr, loss, metrics, timers=None, **kwargs):
        main_log_line = 'Step {} / {}'.format(step, total_steps)
        if lr is not None:
            main_log_line += ', lr: {}'.format(self._format_float(lr))
        if loss is not None:
            main_log_line += ', loss: {}'.format(self._format_float(loss))
        log_lines = [main_log_line, ]
        metrics_log = []
        for k, v in sorted(metrics.items()):
            metrics_log.append(self._format_value(k, v))
        metrics_log = self._soft_wrap(self.delimiter.join(metrics_log))
        log_lines.extend(metrics_log)

        if timers:
            if 'total' in timers:
                log_lines.append(self._estimate_time(timers['total'], step, total_steps))
            timers_log = []
            for k, v in sorted(timers.items()):
                timers_log.append(self._format_value(k, v.average_time))
            timers_log = self._soft_wrap(self.delimiter.join(timers_log))
            log_lines.extend(timers_log)
        self.logger.info('\n  '.join(log_lines))


class TensorboardLogger(TrainingLogger):
    def __init__(self, log_dir, *args, **kwargs):
        self.summary_writer = SummaryWriter(log_dir)

    def close(self):
        self.summary_writer.close()

    def __call__(self, step, total_steps, lr, loss, metrics, timers=None, **kwargs):
        if lr is not None:
            self.summary_writer.add_scalar('learning_rate', lr, step)
        for k, v in metrics.items():
            self.summary_writer.add_scalar(k, v, step)
        if timers:
            for k, v in timers.items():
                self.summary_writer.add_scalar('timers/' + k, v.smoothed_time, step)
