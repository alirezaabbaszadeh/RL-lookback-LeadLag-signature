from leadlag.research.offline_rl.log_trajectories import *  # noqa


def main(argv=None):  # noqa: D401
    """Deprecated wrapper entrypoint. Use 'leadlag.research.offline_rl.log_trajectories:main'."""
    try:
        from leadlag.utils.deprecations import warn_entrypoint_deprecated
        warn_entrypoint_deprecated(
            "research.offline_rl.log_trajectories",
            replacement="leadlag.research.offline_rl.log_trajectories:main",
            remove_in="0.2.0",
        )
    except Exception:
        pass
    from leadlag.research.offline_rl.log_trajectories import main as _main
    return _main(argv)
