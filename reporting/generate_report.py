from leadlag.reporting.generate_report import *  # noqa


def main(argv=None):  # noqa: D401
    """Deprecated wrapper entrypoint. Use 'leadlag.reporting.generate_report:main'."""
    try:
        from leadlag.utils.deprecations import warn_entrypoint_deprecated
        warn_entrypoint_deprecated(
            "reporting.generate_report",
            replacement="leadlag.reporting.generate_report:main",
            remove_in="0.2.0",
        )
    except Exception:
        pass
    from leadlag.reporting.generate_report import main as _main
    return _main()
