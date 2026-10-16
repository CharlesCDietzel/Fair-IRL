__all__ = ["main"]


def __getattr__(name):
    # Imported on first use rather than here: the experiment script pulls in
    # PyTorch and every model library, close to 1 GB, which lightweight
    # submodules such as `fair_irl.config` and `fair_irl.sweep` would otherwise
    # load too.
    if name == "main":
        from fair_irl.Fair_IRL_Biased_Demonstrations import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
