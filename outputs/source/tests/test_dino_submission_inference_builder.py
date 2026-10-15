import ast
import json
from pathlib import Path

import pytest


def test_builds_hash_locked_f2_only_one_image_inference_package(tmp_path):
    from tools.build_dino_submission_inference import build_package

    root = Path(__file__).resolve().parents[1]
    package, notebook, metadata = build_package(root, tmp_path)

    manifest = json.loads((package / "inference_manifest.json").read_text())
    assert manifest["arm"] == "F2"
    assert manifest["folds"] == [0, 1]
    assert manifest["training"] is False
    assert manifest["one_image_smoke"] is True
    for name, expected in manifest["sha256"].items():
        import hashlib

        assert hashlib.sha256((package / name).read_bytes()).hexdigest() == expected

    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["enable_internet"] is False
    assert metadata["competition_sources"] == ["csiro-biomass"]
    assert metadata["kernel_sources"] == []

    cells = json.loads(notebook.read_text())["cells"]
    code = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
    for cell in cells:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    assert set(manifest["sha256"]) == {"fold_0_F2_selected.pt", "fold_1_F2_selected.pt"}
    assert "fold_{fold}_F2_selected.pt" in code
    assert "len(image_paths) != 1" in code
    assert "os.walk('/kaggle/input', followlinks=True)" not in code
    assert code.count("os.walk('/kaggle/input'") <= 1
    assert "STAGE_START" in code
    assert "STAGE_DONE" in code
    assert "INFERENCE_SECONDS_BY_FOLD" in code
    assert "model_provenance['checkpoint_sha256']" in code
    assert "digest(base_checkpoint)" not in code
    assert "name.startswith('timm-1.0.30-')" in code
    assert "optimizer.step" not in code
    assert "fit_arm(" not in code
    assert "submission.csv" in code


def test_refuses_missing_or_changed_selected_checkpoint(tmp_path):
    from tools.build_dino_submission_inference import build_package

    root = Path(__file__).resolve().parents[1]
    package, _, _ = build_package(root, tmp_path)
    (package / "fold_0_F2_selected.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checkpoint hash"):
        build_package(root, tmp_path)


def test_one_image_runner_cannot_submit_partial_competition_csv():
    root = Path(__file__).resolve().parents[1]
    source = (root / "tools/run_dino_submission_inference.py").read_text()
    assert "competition_submit_code" not in source
    assert '"competition_submitted": False' in source
    assert "one-image diagnostic output is not a full competition submission" in source


def test_full_inference_notebook_handles_every_test_image(tmp_path):
    from tools.build_dino_submission_inference import build_package

    root = Path(__file__).resolve().parents[1]
    package, notebook, metadata = build_package(
        root, tmp_path / "weights", one_image_smoke=False)
    cells = json.loads(notebook.read_text())["cells"]
    code = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
    ast.parse(code)
    assert metadata["id"] == "meghmayur1186/img2biomass-dino-f2-full-inference-v4"
    assert "for image_path in tqdm(image_paths" in code
    assert "len(submission) != len(test)" in code
    assert "'all_test_images'" in code


def test_exact_submitted_v4_maps_multiple_images_to_all_requested_rows(tmp_path):
    """Execute the submitted orchestration with model inference replaced by a stub."""
    import numpy as np
    import pandas as pd
    import time
    from types import SimpleNamespace

    root = Path(__file__).resolve().parents[1]
    notebook = root / "outputs/dino_f2_competition_inference/full_inference/source/dino_f2_all_image_t4_inference.ipynb"
    code = "\n".join("".join(c["source"]) for c in json.loads(notebook.read_text())["cells"]
                     if c["cell_type"] == "code")
    targets = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
    # Reordered CSV rows and repeated image paths emulate hidden-data evaluation.
    test = pd.DataFrame([
        {'sample_id': f'{image}_{target}', 'image_path': f'test/{image}.jpg', 'target_name': target}
        for target in reversed(targets) for image in (3, 1, 2)
    ])
    calls = []
    class Views:
        def __init__(self, value):
            self.value = value
        def unsqueeze(self, axis):
            return self
    def predict(fold, views):
        calls.append((fold, views.value))
        return np.arange(5, dtype=float) + 10 * views.value + fold, 0.01
    namespace = {
        'test': test, 'image_paths': test.image_path.drop_duplicates().tolist(),
        'np': np, 'pd': pd, 'json': json, 'time': time,
        'tqdm': lambda iterable, **kwargs: iterable,
        'TEST_CSV': tmp_path / 'test.csv', 'WORK': tmp_path,
        'manifest': {'folds': [0, 1]}, 'predict_fold': predict,
        'native_views': lambda path: [int(path.stem)],
        'preprocess': lambda tile, size: tile,
        'torch': SimpleNamespace(stack=lambda tiles: Views(tiles[0])),
    }
    section = code[code.index('_inference_started ='):code.index('if len(image_paths) == 1:')]
    exec(compile(section, str(notebook), 'exec'), namespace)
    result = pd.read_csv(tmp_path / 'submission.csv')
    assert list(result.sample_id) == list(test.sample_id)
    assert len(result) == 15 and len(calls) == 6
    expected = [10 * int(Path(row.image_path).stem) + targets.index(row.target_name) + 0.5
                for row in test.itertuples()]
    np.testing.assert_array_equal(result.target, expected)


def test_diagnostic_guard_reproduces_hidden_multi_image_failure():
    from tools.build_dino_submission_inference import _notebook_source
    source = _notebook_source(b'', 'test', one_image_smoke=True)
    guard = source[source.index('if len(image_paths) != 1:'):source.index('if not image_paths:')]
    with pytest.raises(ValueError, match='exactly one visible test image'):
        exec(guard, {'image_paths': ['test/a.jpg', 'test/b.jpg']})
