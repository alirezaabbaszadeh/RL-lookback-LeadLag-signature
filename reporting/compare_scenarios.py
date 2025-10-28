from leadlag.reporting.compare_scenarios import *  # noqa


def main(argv=None):  # noqa: D401
    """Deprecated wrapper entrypoint. Use 'leadlag.reporti ng.compare_scenarios:main'."""
    try:
        from leadlag.utils.deprecations import warn_entrypoint_deprecated
        warn_entrypoint_deprecated(
            "reporting.compare_scenarios",
            replacement="leadlag.reporting.compare_scenarios:main",
            remove_in="0.2.0",
        )
    except Exception:
        pass
    from leadlag.reporting.compare_scenarios import main as _main
    return _main()
