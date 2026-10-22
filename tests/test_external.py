from pathlib import Path

import polars as pl
import pytest

from retina_dr.external import (
    build_external,
    read_deepdrid,
    read_deepdrid_uwf,
    read_idrid,
    read_messidor2,
)
from retina_dr.quality import GOOD, REJECT
from tests.conftest import write_fundus


def _idrid(root: Path) -> Path:
    grading = root / "B. Disease Grading"
    for split, names in (
        ("b. Testing Set", ["IDRiD_001", "IDRiD_002"]),
        ("a. Training Set", ["IDRiD_101"]),
    ):
        for i, name in enumerate(names):
            write_fundus(grading / "1. Original Images" / split / f"{name}.jpg", seed=i)
    truth = grading / "2. Groundtruths"
    truth.mkdir(parents=True)
    (truth / "b. IDRiD_Disease Grading_Testing Labels.csv").write_text(
        "Image name,Retinopathy grade,Risk of macular edema \nIDRiD_001,0,0\nIDRiD_002,3,2\n"
    )
    (truth / "a. IDRiD_Disease Grading_Training Labels.csv").write_text(
        "Image name,Retinopathy grade,Risk of macular edema \nIDRiD_001,4,2\n"
    )
    # Other IDRiD parts reuse the same image names for different files.
    write_fundus(root / "A. Segmentation" / "1. Original Images" / "IDRiD_001.jpg", seed=9)
    return root


def _deepdrid(root: Path) -> Path:
    folder = root / "regular_fundus_images" / "Online-Challenge1&2-Evaluation"
    for i, name in enumerate(["347_l1", "347_l2", "347_r1"]):
        write_fundus(folder / "347" / f"{name}.jpg", seed=i)
    pl.DataFrame({"image_id": ["347_l1", "347_l2", "347_r1"], "DR_Levels": [1, 1, 4]}).write_excel(
        folder / "Challenge1_labels.xlsx"
    )
    pl.DataFrame(
        {"image_id": ["347_l1", "347_l2", "347_r1"], "Overall quality": [1, 0, 1]}
    ).write_excel(folder / "Challenge2_labels.xlsx")
    return root


def _deepdrid_uwf(root: Path) -> Path:
    base = root / "ultra-widefield_images"
    header = "patient_id,image_id,image_path,DR_level,position\n"
    for split, rows in (
        ("ultra-widefield-training", ["1,1_r1,x,0,right_eye", "1,1_l2,x,5,left_eye"]),
        ("ultra-widefield-validation", ["34,34_l2,x,3,left_eye"]),
    ):
        folder = base / split
        folder.mkdir(parents=True)
        (folder / f"{split}.csv").write_text(header + "\n".join(rows) + "\n")
        for row in rows:
            patient, image_id = row.split(",")[:2]
            write_fundus(folder / "Images" / patient / f"{image_id}.jpg", seed=len(image_id))
    evaluation = base / "Online-Challenge3-Evaluation"
    write_fundus(evaluation / "Images" / "58" / "58_l2.jpg", seed=3)
    pl.DataFrame({"image_id": ["58_l2"], "UWF_DR_Levels": [2]}).write_excel(
        evaluation / "Challenge3_labels.xlsx"
    )
    return root


def _messidor2(root: Path) -> Path:
    for i, name in enumerate(["a.png", "b.png"]):
        write_fundus(root / "IMAGES" / name, seed=i)
    (root / "messidor_data.csv").write_text(
        "image_id,adjudicated_dr_grade,adjudicated_dme,adjudicated_gradable\na.png,2,0,1\nb.png,,,0\n"
    )
    return root


def test_read_idrid_keeps_only_the_test_split(tmp_path: Path) -> None:
    records = read_idrid(_idrid(tmp_path / "idrid"))
    assert sorted((r.image_id, r.grade) for r in records) == [("IDRiD_001", 0), ("IDRiD_002", 3)]
    assert all(r.eye_id == r.image_id for r in records)


def test_read_deepdrid_links_images_to_eyes_and_quality(tmp_path: Path) -> None:
    records = {r.image_id: r for r in read_deepdrid(_deepdrid(tmp_path / "deepdrid"))}
    assert records["347_l1"].eye_id == records["347_l2"].eye_id == "347_l"
    assert records["347_r1"].patient_id == "347"
    assert records["347_l1"].quality == GOOD
    assert records["347_l2"].quality == REJECT
    assert records["347_r1"].grade == 4


def test_read_deepdrid_uwf_reads_all_three_splits_and_marks_grade_5(tmp_path: Path) -> None:
    records = {r.image_id: r for r in read_deepdrid_uwf(_deepdrid_uwf(tmp_path / "uwf"))}
    assert set(records) == {"1_r1", "1_l2", "34_l2", "58_l2"}
    assert records["1_l2"].grade is None
    assert records["1_l2"].quality == REJECT
    assert records["34_l2"].grade == 3
    assert records["58_l2"].grade == 2
    assert records["58_l2"].patient_id == "58"
    assert records["1_r1"].eye_id == "1_r"
    assert records["1_r1"].source == "deepdrid_uwf"


def test_read_messidor2_marks_ungradable_images(tmp_path: Path) -> None:
    records = {r.image_id: r for r in read_messidor2(_messidor2(tmp_path / "messidor2"))}
    assert records["a"].grade == 2
    assert records["a"].quality == GOOD
    assert records["b"].grade is None
    assert records["b"].quality == REJECT


def test_build_external_writes_one_manifest_for_the_sources_present(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _idrid(raw / "idrid")
    _messidor2(raw / "messidor2")
    out = tmp_path / "external"
    manifest = build_external(raw, out, size=32)
    assert set(manifest["source"].unique()) == {"idrid", "messidor2"}
    assert (manifest["split"] == "external").all()
    ungradable = manifest.filter(pl.col("image_id") == "b")
    assert ungradable["binary_label"].to_list() == [None]
    assert all((out / p).is_file() for p in manifest["path"].to_list())
    assert pl.read_parquet(out / "manifest.parquet").height == manifest.height


def test_build_external_fails_when_no_source_is_present(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_external(tmp_path / "empty", tmp_path / "out", size=32)


def test_score_external_applies_frozen_thresholds_per_eye(tmp_path: Path) -> None:
    import json

    from retina_dr.train.engine import TrainConfig, train
    from retina_dr.train.external_eval import score_external
    from retina_dr.train.report import ensemble_runs
    from tests.test_training import TINY_MODEL, _processed_dir

    data = _processed_dir(tmp_path / "data")
    runs = []
    for name, task in (("dr", "dr"), ("q", "quality")):
        config = TrainConfig(
            data_dir=data,
            run_dir=tmp_path / name,
            model=TINY_MODEL,
            pretrained=False,
            image_size=32,
            epochs=1,
            batch_size=4,
            workers=0,
            device="cpu",
            task=task,
            tta=False,
        )
        runs.append(train(config))
    ensemble = tmp_path / "ens"
    ensemble_runs([runs[0]], ensemble, n_resamples=0, quality_run=runs[1])

    raw = tmp_path / "raw"
    _messidor2(raw / "messidor2")
    _deepdrid(raw / "deepdrid")
    external = tmp_path / "external"
    build_external(raw, external, size=32)

    out = tmp_path / "scored"
    score_external(
        external, [runs[0]], runs[1], ensemble, out, device="cpu", tta=False, n_resamples=0
    )
    predictions = pl.read_parquet(out / "predictions.parquet")
    assert predictions.height == 5
    metrics = json.loads((out / "metrics.json").read_text())
    frozen = json.loads((ensemble / "metrics.json").read_text())
    assert metrics["threshold"] == frozen["threshold"]
    deepdrid = next(r for r in metrics["rows"] if r["source"] == "deepdrid")
    assert deepdrid["images"] == 2  # two eyes from three images
    assert metrics["ungradable"]["messidor2"]["eyes"] == 1


def test_load_checkpoint_accepts_runs_trained_before_the_quality_head(tmp_path: Path) -> None:
    import torch

    from retina_dr.train.external_eval import load_checkpoint
    from retina_dr.train.model import build_model
    from tests.test_training import TINY_MODEL

    model = build_model(TINY_MODEL, pretrained=False)
    state = {k: v for k, v in model.state_dict().items() if not k.startswith("quality_head.")}
    torch.save({"model": state}, tmp_path / "best.pt")
    fresh = build_model(TINY_MODEL, pretrained=False)
    load_checkpoint(fresh, tmp_path / "best.pt", torch.device("cpu"))
    assert torch.equal(fresh.binary_head.weight, model.binary_head.weight)


def test_load_checkpoint_rejects_other_missing_weights(tmp_path: Path) -> None:
    import torch

    from retina_dr.train.external_eval import load_checkpoint
    from retina_dr.train.model import build_model
    from tests.test_training import TINY_MODEL

    model = build_model(TINY_MODEL, pretrained=False)
    state = {k: v for k, v in model.state_dict().items() if not k.startswith("binary_head.")}
    torch.save({"model": state}, tmp_path / "best.pt")
    with pytest.raises(RuntimeError, match="binary_head"):
        load_checkpoint(
            build_model(TINY_MODEL, pretrained=False), tmp_path / "best.pt", torch.device("cpu")
        )
