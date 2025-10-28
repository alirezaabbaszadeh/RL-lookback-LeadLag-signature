from leadlag.pipelines.run_full_suite import *  # noqa


def main(argv=None):  # noqa: D401
    """Deprecated wrapper entrypoint. Use 'leadlag.pipelines.run_full_suite:main'."""
    try:
        from leadlag.utils.deprecations import warn_entrypoint_deprecated
        warn_entrypoint_deprecated(
            "pipelines.run_full_suite",
            replacement="leadlag.pipelines.run_full_suite:main",
            remove_in="0.2.0",
        )
    except Exception:
        pass
    from leadlag.pipelines.run_full_suite import main as _main
    return _main(argv)
