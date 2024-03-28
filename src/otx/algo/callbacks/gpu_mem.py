from lightning.pytorch.callbacks import DeviceStatsMonitor


class OTXMemMonitor(DeviceStatsMonitor):
    def _get_and_log_device_stats(self, trainer: "pl.Trainer", key: str) -> None:
        if not trainer._logger_connector.should_update_logs:
            return

        device = trainer.strategy.root_device
        if self._cpu_stats is False and device.type == "cpu":
            # cpu stats are disabled
            return

        device_stats = trainer.accelerator.get_device_stats(device)

        if self._cpu_stats and device.type != "cpu":
            # Don't query CPU stats twice if CPU is accelerator
            from lightning.pytorch.accelerators.cpu import get_cpu_stats

            device_stats.update(get_cpu_stats())

        for logger in trainer.loggers:
            peak_mem_gb = device_stats['active_bytes.all.peak'] * 1e-9
            peak_mem = {f"{str(device)}.{key}": peak_mem_gb}
            logger.log_metrics(peak_mem, step=trainer.fit_loop.epoch_loop._batches_that_stepped)