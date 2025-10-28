from leadlag.hydra_main import *  # noqa


def main(argv=None):  # noqa: D401
    """Deprecated wrapper entrypoint. Use 'leadlag.hydra_main:main'."""
    try:
        from leadlag.utils.deprecations import warn_entrypoint_deprecated
        warn_entrypoint_deprecated(
            "hydra_main.py",
            replacement="leadlag.hydra_main:main",
            remove_in="0.2.0",
        )
    except Exception:
        pass
    from leadlag.hydra_main import main as _main
    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
