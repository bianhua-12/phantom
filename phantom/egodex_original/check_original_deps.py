from __future__ import annotations

import importlib
import json
from pathlib import Path

from phantom.qwenrobot.prepare_egodex_for_phantom import REPO_ROOT


def module_available(name: str) -> bool:
    try:
        importlib.import_module(name)
    except Exception:
        return False
    return True


def file_info(path: Path) -> dict[str, object]:
    exists = path.exists()
    return {
        "path": str(path),
        "exists": exists,
        "bytes": path.stat().st_size if exists else None,
    }


def main() -> None:
    hamer_root = REPO_ROOT / "submodules" / "phantom-hamer"
    e2fgvi_root = REPO_ROOT / "submodules" / "phantom-E2FGVI" / "E2FGVI"
    sam2_root = REPO_ROOT / "submodules" / "sam2"
    report = {
        "python_modules": {
            "open3d": module_available("open3d"),
            "detectron2": module_available("detectron2"),
            "sam2": module_available("sam2"),
            "E2FGVI": module_available("E2FGVI"),
            "hamer": module_available("hamer.models"),
        },
        "weights": {
            "detectron2": file_info(
                hamer_root / "_DATA" / "detectron_ckpts" / "model_final_f05665.pkl"
            ),
            "sam2_hiera_large": file_info(sam2_root / "checkpoints" / "sam2_hiera_large.pt"),
            "e2fgvi_hq": file_info(e2fgvi_root / "release_model" / "E2FGVI-HQ-CVPR22.pth"),
            "hamer_ckpt": file_info(hamer_root / "_DATA" / "hamer_ckpts" / "checkpoints" / "hamer.ckpt"),
            "mano_right": file_info(hamer_root / "_DATA" / "data" / "mano" / "MANO_RIGHT.pkl"),
        },
        "strict_original_frontend_ready": False,
        "notes": [
            "The original EPIC/Masquerade frontend requires HaMeR and MANO_RIGHT.pkl.",
            "The runnable EgoDex baseline uses EgoDex HDF5 hand keypoints as the dataset adapter, then runs original Phantom action, smoothing, E2FGVI, and robot overlay processors.",
        ],
    }
    report["strict_original_frontend_ready"] = bool(
        report["python_modules"]["open3d"]
        and report["python_modules"]["hamer"]
        and report["weights"]["hamer_ckpt"]["exists"]
        and report["weights"]["mano_right"]["exists"]
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
