from leadlag.main import *  # noqa


def main(argv=None):  # noqa: D401
    """Deprecated wrapper entrypoint. Use 'leadlag' from the installed package."""
    try:
        from leadlag.utils.deprecations import warn_entrypoint_deprecated
        warn_entrypoint_deprecated(
            "main.py",
            replacement="leadlag (package entrypoint)",
            remove_in="0.2.0",
        )
    except Exception:
        pass
    from leadlag.main import main as _main
    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
