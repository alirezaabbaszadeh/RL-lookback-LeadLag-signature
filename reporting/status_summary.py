from leadlag.reporting.status_summary import *  # noqa


def main(argv=None):  # noqa: D401
    """Deprecated wrapper entrypoint. Use 'leadlag.reporting.status_summary:main'."""
    try:
        from leadlag.utils.deprecations import warn_entrypoint_deprecated
        warn_entrypoint_deprecated(
            "reporting.status_summary",
            replacement="leadlag.reporting.status_summary:main",
            remove_in="0.2.0",
        )
    except Exception:
        pass
    from leadlag.reporting.status_summary import main as _main
    return _main()
