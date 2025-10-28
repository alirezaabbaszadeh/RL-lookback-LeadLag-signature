from leadlag.reporting.plot_balance_history import *  # noqa


def main(argv=None):  # noqa: D401
    """Deprecated wrapper entrypoint. Use 'leadlag.reporting.plot_balance_history:main'."""
    try:
        from leadlag.utils.deprecations import warn_entrypoint_deprecated
        warn_entrypoint_deprecated(
            "reporting.plot_balance_history",
            replacement="leadlag.reporting.plot_balance_history:main",
            remove_in="0.2.0",
        )
    except Exception:
        pass
    from leadlag.reporting.plot_balance_history import main as _main
    return _main()
