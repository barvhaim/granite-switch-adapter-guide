import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    "00_macos_mps_preflight.ipynb",
    "01_prepare_and_validate_data.ipynb",
    "02_train_lora_on_macos.ipynb",
    "03_train_alora_on_macos.ipynb",
    "04_evaluate_package_and_compose.ipynb",
]


def load_notebook(name: str) -> dict:
    return json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))


def source_text(cell: dict) -> str:
    source = cell.get("source", [])
    return source if isinstance(source, str) else "".join(source)


def test_expected_notebooks_exist_and_are_valid_v4():
    notebook_dir = ROOT / "notebooks"
    assert sorted(path.name for path in notebook_dir.glob("*.ipynb")) == NOTEBOOKS
    for name in NOTEBOOKS:
        notebook = load_notebook(name)
        assert notebook["nbformat"] == 4
        assert notebook["nbformat_minor"] >= 5
        assert notebook["metadata"]["kernelspec"]["language"] == "python"
        assert notebook["cells"]
        assert notebook["cells"][0]["cell_type"] == "markdown"


def test_notebook_python_cells_compile():
    for name in NOTEBOOKS:
        for index, cell in enumerate(load_notebook(name)["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = source_text(cell)
            assert not source.lstrip().startswith(("!", "%")), (
                f"{name} cell {index} uses shell or line magic; use Python APIs for portability"
            )
            ast.parse(source, filename=f"{name}:cell-{index}")


def test_macos_training_notebooks_contain_safety_and_mps_contracts():
    preflight = json.dumps(load_notebook(NOTEBOOKS[0]))
    assert "torch.backends.mps.is_available" in preflight
    assert "torch.mps.synchronize" in preflight
    assert "64" in preflight

    for name, technology in ((NOTEBOOKS[2], "lora"), (NOTEBOOKS[3], "alora")):
        text = json.dumps(load_notebook(name))
        assert "RUN_TRAINING = False" in text
        assert f'"{technology}"' in text
        assert "--gradient-checkpointing" in text
        assert "mps" in text.lower()


def test_notebooks_do_not_embed_execution_outputs():
    for name in NOTEBOOKS:
        for cell in load_notebook(name)["cells"]:
            if cell["cell_type"] == "code":
                assert cell.get("execution_count") is None
                assert cell.get("outputs") == []
