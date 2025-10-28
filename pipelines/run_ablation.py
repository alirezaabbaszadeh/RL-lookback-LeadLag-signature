from leadlag.pipelines.run_ablation import *  # noqa


def main(argv=None):  # noqa: D401
    """Deprecated wrapper entrypoint. Use 'leadlag.pipelines.run_ablation:main'."""
    try:
        from leadlag.utils.deprecations import warn_entrypoint_deprecated
        warn_entrypoint_deprecated(
            "pipelines.run_ablation",
            replacement="leadlag.pipelines.run_ablation:main",
            remove_in="0.2.0",
        )
    except Exception:
        pass
    from leadlag.pipelines.run_ablation import main as _main
    return _main(argv)
