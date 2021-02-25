import csv
import re
import sys
import time
from collections import OrderedDict
from contextlib import contextmanager

from .utils import AverageMeter


class Handler:
    """Observes events from TrainingLogger and dispatches them to a specific destination."""

    def write(self, value):
        """Write value to handler.

        Args:
            value (Value): currently logged value.
        """
        pass

    def start(self, scope, step, total):
        """Indicate that scope is started"""
        pass

    def end(self, scope):
        """Indicate that scope is finished"""
        pass


class PeriodicHandler(Handler):
    """Handler that flushes written values when scope ends (e.g. epoch ends)."""

    def __init__(self, scope):
        super().__init__()
        self.values = OrderedDict()
        self.total = 0
        self.step = 0
        self.epoch = 0
        self.scope = scope

    def write(self, value):
        self.values[value.tag] = value

        if self.scope == 'global':
            self.end(self.scope)
            self.start(self.scope, self.step + 1, None)

    def end(self, scope):
        if scope == self.scope:
            self.flush()

    def start(self, scope, step, total, epoch=0):
        if scope != self.scope:
            return
        if step is None:
            step = self.step + 1
        self.step = step
        self.epoch = epoch
        self.total = total
        self.values.clear()

    def flush(self):
        """Writes saved values"""
        raise NotImplementedError


class StreamHandler(PeriodicHandler):
    """Writes logged values either to stream or python logger. """

    def __init__(self, scope, prefix='', fmt="{prefix}[{epoch}][{step}/{total}]\t{values}", display_instant=True,
                 display_frequency=1, stream=None, logger=None):
        super().__init__(scope)
        if stream is None:
            stream = sys.stdout
        self.logger = logger
        self.stream = stream
        self.display_frequency = display_frequency
        self.display_instant = display_instant
        self.fmt = fmt
        self.prefix = prefix

        self._display_count = 0

    def end(self, scope):
        if scope == self.scope:
            self._display_count += 1
            if self._display_count == self.display_frequency:
                if self.values:
                    self.flush()
                self._display_count = 0

    def flush(self):
        msg = self.format_msg()
        if self.logger:
            self.logger.info(msg)
        else:
            self.stream.write(msg + '\n')
            self.stream.flush()

    def format_value(self, value):
        if value.instant_value is not None and self.display_instant:
            return "{name} {ival:.4f} ({val:.4f})".format(name=value.display_name, val=value.value,
                                                          ival=value.instant_value)
        else:
            return "{name} {val:.4f}".format(name=value.display_name, val=value.value)

    def format_msg(self):
        values_fmt = [self.format_value(v) for v in self.values.values()]
        values_s = "\t".join(values_fmt)
        msg = self.fmt.format(prefix=self.prefix, epoch=self.epoch, step=self.step, total=self.total, values=values_s)
        return msg


class TensorboardHandler(PeriodicHandler):
    """Writes logged values to tensorboard."""

    def __init__(self, scope, summary_writer):
        super().__init__(scope)
        self.summary_writer = summary_writer

    def flush(self):
        for tag, value in self.values.items():
            step = self.step if self.scope != 'global' else value.step
            self.summary_writer.add_scalar(tag, value.value, step)


class CSVHandler(PeriodicHandler):
    """Writes logged values to CSV file"""

    def __init__(self, scope, csv_path, index_col='step'):
        super().__init__(scope)
        self.index_col = index_col
        self.csv_path = str(csv_path)
        self._csv_file = open(self.csv_path, 'w')
        self.csv_writer = csv.writer(self._csv_file, delimiter='\t')
        self._header = None

    def flush(self):
        if not self.values:
            return
        if self._header is None:
            self._header = [self.index_col] + [value.display_name for value in self.values.values()]
            self.csv_writer.writerow(self._header)

        new_row = [self.step if self.scope != 'global' else next(iter(self.values.values())).step]

        for value_obj in self.values.values():
            new_row.append(value_obj.value)

        self.csv_writer.writerow(new_row)
        self._csv_file.flush()


class Value:
    """Wraps logged value and maintain accumulated version."""

    def __init__(self, tag, handlers, aggregator=None, save_instant=True, display_name=None):
        if display_name is None:
            display_name = tag

        self.tag = tag
        self.value = None
        self.instant_value = None
        self.step = 0
        self.save_instant = save_instant
        self.handlers = handlers
        self.aggregator = aggregator
        self.display_name = display_name

    def reset(self):
        """Reset aggregation and step counter"""
        if self.aggregator:
            self.aggregator.reset()
        self.step = 0

    def update(self, value, num_averaged=1, step=None):
        if step is None:
            step = self.step + 1
        self.step = step
        if self.save_instant:
            self.instant_value = value
        self.value = value
        if self.aggregator:
            self.aggregator.update(value, num_averaged)
            self.value = self.aggregator.avg


class TrainingLogger:
    """Logging utility that accumulates all logged values within some scope (e.g. batch / epoch / etc),
    and dispatches them to possibly multiple handlers.


    Logged values should be registered with register_value or register_value_group method. Values should be attached
    to handlers that correspond to a specific output stream (e.g. console, csv, TensorBoard, etc), which should be
    registered in advance. After handlers and values are registered, logging can be accomplished by calling log_value
    method. In order to indicate when handler flush logged values (e.g. to format all values, logged during epoch in
    one line), all log_value calls should occur between start scope and end scope.
    """

    def __init__(self):
        self._handlers = OrderedDict()
        self._values = OrderedDict()
        self._value_groups = OrderedDict()

    def log_value(self, tag, value, num_averaged=1, step=None):
        """Dispatch value to its handlers
        Args:
            tag (str): Name of the logged value. Value should be registered with the same name before logging.
            value (float): The actual value (e.g. current batch loss)
            num_averaged: If the value is registered with average=True and num_averaged > 1, then accumulated value
            will be adjusted correctly. Use it in case if the actual value is the average of multiple value.
            step (int): Order of the value (e.g. number of batch)
        """
        value_obj = self._resolve_tag(tag)
        value_obj.update(value, num_averaged, step)
        for handler in value_obj.handlers:
            handler.write(value_obj)

    def register_value(self, tag, handlers, average=False, display_instant=None, display_name=None):
        """Register value and attach it to handlers.

        Args:
            tag (str): Name of the value.
            handlers (List[str]): List of handler names that this value will be attached to.
            average (bool): Whether value should be averaged, i.e. cumulative value will be passed to the handler.
            display_instant (bool): Whether instant (non-cumulative) must be flushed by the handler as well (not every
            handler will interpret this option)
            display_name (str): Readable name of the value
        """
        if display_instant is None:
            display_instant = average
        handlers = [self._handlers[handler] for handler in handlers]
        if average:
            aggregator = AverageMeter()
        else:
            aggregator = None
        value = Value(tag, handlers, aggregator=aggregator, save_instant=display_instant, display_name=display_name)
        self._values[tag] = value

    def register_value_group(self, group_pattern, handlers, average=False, display_instant=None):
        """Register group of values and attach it to the same handler. New values will be registered on the fly.
        Args:
            group_pattern (str): Regular expression of the group name.
            handlers (List[str]): List of handler names that this values will be attached to.
            average (bool): Whether value should be averaged, i.e. cumulative value will be passed to handler.
            display_instant (bool): Whether instant (non cumulative) must be flushed by the handler as well (not every
            handler will interpret this option)
        """
        self._value_groups[group_pattern] = (re.compile(group_pattern), handlers, average, display_instant)

    def register_handler(self, name, handler):
        """Register value handler with a given name"""
        self._handlers[name] = handler

    def start_scope(self, scope, step=None, total=None, epoch=0):
        """Indicate start of the scope to registered handlers

        Args:
            scope (str): Started scope number
            step (int): Started scope number
            total (int): How many scopes with given name there will be in a row. Used by certain handlers to display
            current progress
            epoch (int): Current epoch number

        """
        for handler in self._handlers.values():
            handler.start(scope, step, total, epoch=epoch)

    def end_scope(self, scope):
        """Indicate end of the scope to registered handlers"""
        for handler in self._handlers.values():
            handler.end(scope)

    def start_epoch(self, epoch, total=None):
        """Start "epoch" scope"""
        self.start_scope('epoch', step=epoch, total=total, epoch=epoch)

    def start_batch(self, step=None, total=None, epoch=0):
        """Start "batch" scope"""
        self.start_scope('batch', step=step, total=total, epoch=epoch)

    def end_batch(self):
        """End "batch" scope"""
        self.end_scope(scope='batch')

    def end_epoch(self):
        """End "epoch" scope"""
        self.end_scope(scope='epoch')

    def get_value(self, tag):
        """Returns current cumulative value"""
        return self._values[tag].value

    @contextmanager
    def scope(self, epoch=0, scope='epoch', total=None):
        """Context manager that begins and ends scope with name "scope" """
        self.start_scope(scope, epoch, total, epoch)
        yield
        self.end_scope(scope)

    def reset_values(self, values_pattern):
        """Resets accumulated value of values with given pattern"""
        rx = re.compile(values_pattern)
        for value_name, value in self._values.items():
            if rx.search(value_name):
                value.reset()

    def scope_enumerate(self, iterable, epoch=0, scope='batch', total_time=None, fetch_time=None, body_time=None):
        """The same as enumerate() but wraps every iteration with begin and end scope. Optionally can log cpu time
        for certain code sections

        Args:
            iterable: Object that will be used for iteration
            epoch (int): Current epoch number
            scope (str): Name of the scope that will wrap every iteration.
            total_time (str): Name of the value that will be used for logging total iteration time (if set to any value)
            fetch_time (str): Name of the value that will be used for logging time that iterable takes to produce new
                              item (if set to any value).
            body_time (str): Name of the value that will be used for logging time of executing loop body (if set to
            any value).

        """
        start_time = time.time()

        t0 = time.time()
        total = len(iterable)
        for i, data in enumerate(iterable):
            t1 = time.time()
            self.start_scope(scope, i + 1, total=total, epoch=epoch)
            yield i, data
            t2 = time.time()

            if fetch_time:
                self.log_value(fetch_time, t1 - t0)
            if body_time:
                self.log_value(body_time, t2 - t1)
            self.end_scope(scope)
            t0 = t2
        if total_time:
            self.log_value(total_time, time.time() - start_time)

    def _resolve_tag(self, tag):
        if tag in self._values:
            return self._values[tag]
        for p, (rx, handlers, average, display_instant) in self._value_groups.items():
            if rx.match(tag):
                self.register_value(tag, handlers, average, display_instant, tag)
        return self._values[tag]
