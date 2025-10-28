from leadlag.research.offline_rl.train_offline import *  # noqa


def main(argv=None):  # noqa: D401
    """Deprecated wrapper entrypoint. Use 'leadlag.research.offline_rl.train_offline:main'."""
    try:
        from leadlag.utils.deprecations import warn_entrypoint_deprecated
        warn_entrypoint_deprecated(
            "research.offline_rl.train_offline",
            replacement="leadlag.research.offline_rl.train_offline:main",
            remove_in="0.2.0",
        )
    except Exception:
        pass
    from leadlag.research.offline_rl.train_offline import main as _main
    return _main(argv)
