class CallbackProtocol:
    def on_train_begin(self, logs: dict | None = None) -> None:
        pass

    def on_train_end(self, logs: dict | None = None) -> None:
        pass

    def on_epoch_begin(self, epoch: int, logs: dict | None = None) -> None:
        pass

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        pass

    def on_batch_begin(
        self, batch: int | None = None, logs: dict | None = None
    ) -> None:
        pass

    def on_batch_end(self, batch: int | None = None, logs: dict | None = None) -> None:
        pass

    def on_eval_begin(self, logs: dict | None = None) -> None:
        pass

    def on_eval_end(self, logs: dict | None = None) -> None:
        pass


class CallbackHandler(CallbackProtocol):
    def __init__(self, callbacks: list[CallbackProtocol] | None = None):
        self.callbacks = callbacks if callbacks is not None else []

    def on_train_begin(self, logs: dict | None = None) -> None:
        for callback in self.callbacks:
            if hasattr(callback, "on_train_begin"):
                callback.on_train_begin(logs)

    def on_train_end(self, logs: dict | None = None):
        for callback in self.callbacks:
            if hasattr(callback, "on_train_end"):
                callback.on_train_end(logs)

    def on_epoch_begin(self, epoch: int, **kwargs):
        for callback in self.callbacks:
            if hasattr(callback, "on_epoch_begin"):
                callback.on_epoch_begin(epoch, **kwargs)

    def on_epoch_end(self, epoch: int, logs: dict | None = None):
        for callback in self.callbacks:
            if hasattr(callback, "on_epoch_end"):
                callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch: int | None = None, logs: dict | None = None):
        for callback in self.callbacks:
            if hasattr(callback, "on_batch_begin"):
                callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch: int | None = None, logs: dict | None = None):
        for callback in self.callbacks:
            if hasattr(callback, "on_batch_end"):
                callback.on_batch_end(batch, logs)

    def on_eval_begin(self, **kwargs):
        for callback in self.callbacks:
            if hasattr(callback, "on_eval_begin"):
                callback.on_eval_begin(**kwargs)

    def on_eval_end(self, **kwargs):
        for callback in self.callbacks:
            if hasattr(callback, "on_eval_end"):
                callback.on_eval_end(**kwargs)
