from leadlag.utils.resources import open_text, resolve_path, resolve_text


def test_resolve_text_from_package():
    data = resolve_text("leadlag.utils", "resources.py")
    assert data is not None
    assert "resolve_path" in data


def test_resolve_path_fallback_configs():
    path = resolve_path("leadlag.configs", "scenarios/fixed_30.yaml")
    assert path is not None and path.exists()


def test_open_text_context_manager():
    with open_text("leadlag.utils", "resources.py") as handle:
        contents = handle.read()
    assert "resolve_text" in contents
