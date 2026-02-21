import sys
import tempfile
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from ...core.registry import register_skill
from ...core.base_skill import SkillResult
from ...config.skill_configs import GigaPoseConfig
from .base_pose import BasePoseSkill


@register_skill("gigapose")
class GigaPoseSkill(BasePoseSkill):
    """
    Template-based 6-DoF pose estimation using GigaPose.

    GigaPose matches a query image against pre-rendered templates of the
    target object and estimates the full SE(3) pose from a single RGB view.

    Prerequisites
    -------------
    1. Clone the repo::

           git clone https://github.com/nv-nguyen/gigapose ~/libs/gigapose

    2. Install Python dependencies::

           pip install -e ~/libs/gigapose
           # also requires: pytorch_lightning, hydra-core, omegaconf, einops

    3. Download the pre-trained checkpoint::

           python ~/libs/gigapose/src/scripts/download_gigapose.py
           # default location: ~/libs/gigapose/pretrained/gigaPose_v1.ckpt

    4. Render templates for your object
       (needs Panda3D + a .ply/.obj CAD model)::

           python ~/libs/gigapose/src/scripts/render_custom_templates.py

    Usage::

        skill = SkillRegistry.create("gigapose", {
            "checkpoint":    "~/libs/gigapose/pretrained/gigaPose_v1.ckpt",
            "template_dir":  "/path/to/bop_templates/obj_000001",
            "intrinsics":    [fx, fy, cx, cy],
            "device": "cuda",
        })
        result = skill(
            image,                   # HxWx3 uint8 RGB
            bbox=[x1, y1, x2, y2],  # 2-D detection bounding box (pixels)
            obj_id=1,                # BOP object ID (1-indexed integer)
            mask=mask_uint8,         # HxW uint8 object mask (optional)
        )
        # result.data["pose_matrix"]   np.ndarray [4,4]  SE(3) camera←object
        # result.data["rotation"]      np.ndarray [3,3]
        # result.data["translation"]  np.ndarray [3,]  metres (if K provided)
        # result.data["score"]        float   template-match confidence
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.cfg = GigaPoseConfig(**config)
        self._model = None
        self._template_cache: dict = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _gigapose_root(self) -> Path:
        p = Path(self.cfg.gigapose_dir).expanduser().resolve()
        if not p.exists():
            raise RuntimeError(
                f"GigaPose source not found at {p}.\n"
                "Clone with:\n"
                "  git clone https://github.com/nv-nguyen/gigapose "
                "~/libs/gigapose"
            )
        return p

    def _add_to_sys_path(self, root: Path) -> None:
        # Hydra configs reference modules as 'src.models.gigaPose.*',
        # so the REPO ROOT (not src/) must be on sys.path.
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

    def _check_imports(self) -> None:
        try:
            import src.models.gigaPose  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "GigaPose packages are not importable.  "
                "Ensure all dependencies are installed:\n"
                "  pip install -e ~/libs/gigapose\n"
                "  pip install pytorch_lightning hydra-core omegaconf einops"
            ) from exc

    def _check_checkpoint(self) -> Path:
        if self.cfg.checkpoint is None:
            root = self._gigapose_root()
            default = root / "pretrained" / "gigaPose_v1.ckpt"
            raise FileNotFoundError(
                f"No checkpoint configured.  "
                f"Expected default at:\n  {default}\n"
                "Download with:\n"
                "  python ~/libs/gigapose/src/scripts/download_gigapose.py"
            )
        ckpt = Path(self.cfg.checkpoint).expanduser().resolve()
        if not ckpt.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt}\n"
                "Download with:\n"
                "  python ~/libs/gigapose/src/scripts/download_gigapose.py"
            )
        return ckpt

    # ------------------------------------------------------------------
    # BaseSkill lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        import torch
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from hydra.utils import instantiate

        root = self._gigapose_root()
        self._add_to_sys_path(root)
        self._check_imports()
        ckpt_path = self._check_checkpoint()

        # Initialize Hydra (reset any previous state so this is idempotent)
        GlobalHydra.instance().clear()
        with initialize_config_dir(
            config_dir=str(root / "configs"), version_base=None
        ):
            cfg = compose(
                config_name="test",
                overrides=[
                    f"machine.root_dir={root}",
                    "model=large",
                ],
            )

        # Instantiate model from Hydra config
        self._model = instantiate(cfg.model)

        # Load pre-trained weights
        ckpt = torch.load(str(ckpt_path), map_location=self.cfg.device)
        state_dict = ckpt.get("state_dict", ckpt)
        missing, unexpected = self._model.load_state_dict(
            state_dict, strict=False
        )
        if missing:
            self.logger.warning(
                f"Missing keys in checkpoint: {missing[:5]}…"
            )

        self._model.to(self.cfg.device)
        self._model.eval()

        # Give the model a temporary log_dir for any internal saves
        self._model.log_dir = tempfile.mkdtemp(prefix="gigapose_")

        self.logger.info(
            f"GigaPose loaded | device={self.cfg.device}"
            f" | ckpt={ckpt_path.name}"
        )

    def preprocess(
        self,
        image: np.ndarray,
        bbox: Optional[list] = None,
        obj_id: int = 1,
        mask: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Tuple:
        if self.cfg.template_dir is None:
            raise ValueError(
                "template_dir must be set.  Point it to the BOP-format "
                "template directory for the target object.\n"
                "Render with:\n"
                "  python ~/libs/gigapose/src/scripts/"
                "render_custom_templates.py"
            )
        tdir = Path(self.cfg.template_dir).expanduser().resolve()
        if not tdir.exists():
            raise FileNotFoundError(f"template_dir not found: {tdir}")

        h, w = image.shape[:2]
        if self.cfg.intrinsics is not None:
            fx, fy, cx, cy = self.cfg.intrinsics
        else:
            f = float(max(h, w))
            fx = fy = f
            cx, cy = w / 2.0, h / 2.0
            self.logger.warning(
                "No camera intrinsics provided — using rough estimate. "
                "Pass intrinsics=[fx,fy,cx,cy] for metric accuracy."
            )
        K = np.array(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32
        )

        if bbox is None:
            bbox = [0, 0, w, h]
        if mask is None:
            mask = np.ones((h, w), dtype=np.uint8) * 255

        return image, mask, K, [int(v) for v in bbox], obj_id, str(tdir)

    def infer(self, processed: Tuple, **kwargs):
        import torch
        import pandas as pd
        import src.megapose.utils.tensor_collection as tc
        from torchvision.transforms.functional import to_tensor
        import cv2

        image, mask, K, bbox, obj_id, template_dir = processed
        device = self.cfg.device
        h_img, w_img = image.shape[:2]

        # 1. Load / cache template features
        cache_key = f"{template_dir}::{obj_id}"
        if cache_key not in self._template_cache:
            self._load_templates(template_dir, obj_id, cache_key)

        # 2. Crop query image to bbox
        model_size = 224  # GigaPose default input resolution
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)
        bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)

        crop_rgb = cv2.resize(image[y1:y2, x1:x2], (model_size, model_size))
        crop_msk = cv2.resize(mask[y1:y2, x1:x2], (model_size, model_size))

        # Affine from full image to crop (scale + translate)
        sx = model_size / bw
        sy = model_size / bh
        M_np = np.array(
            [[sx, 0, -x1 * sx], [0, sy, -y1 * sy], [0, 0, 1]],
            dtype=np.float32,
        )

        tar_img = to_tensor(crop_rgb).unsqueeze(0).to(device)   # [1,3,H,W]
        tar_mask = torch.from_numpy(
            (crop_msk > 0).astype(np.float32)
        ).unsqueeze(0).to(device)                                # [1,H,W]
        tar_K = torch.from_numpy(K).unsqueeze(0).to(device)     # [1,3,3]
        tar_M = torch.from_numpy(M_np).unsqueeze(0).to(device)  # [1,3,3]

        infos = pd.DataFrame(
            {"label": [obj_id], "scene_id": [0], "view_id": [0]}
        )
        batch = tc.PandasTensorCollection(
            infos=infos,
            tar_img=tar_img,
            tar_mask=tar_mask,
            tar_K=tar_K,
            tar_M=tar_M,
        )

        # 3. Run template matching + pose recovery
        with torch.no_grad():
            template_data = self._template_cache[cache_key]["data"]
            pose_recovery = self._template_cache[cache_key]["recovery"]

            tar_label = torch.tensor([obj_id], dtype=torch.long, device=device)

            # AE feature extraction
            tar_ae = self._model.ae_net(batch.tar_img)
            src_ae = template_data.ae_features[tar_label - 1]
            src_masks = template_data.mask[tar_label - 1]

            # Nearest-neighbour template search
            preds = self._model.testing_metric.test(
                src_feats=src_ae,
                tar_feat=tar_ae,
                src_masks=src_masks,
                tar_mask=batch.tar_mask,
                max_batch_size=None,
            )
            preds.infos = infos

            # IST: predict relative scale and in-plane rotation
            k = self._model.testing_metric.k
            num_patches = preds.src_pts.shape[2]
            src_ist = template_data.ist_features[tar_label - 1]
            tar_ist = self._model.ist_net.forward_by_chunk(batch.tar_img)

            pred_scales = torch.zeros(1, k, num_patches, device=device)
            pred_cosSin = torch.zeros(1, k, num_patches, 2, device=device)
            idx_s = torch.arange(1, device=device)

            for idx_k in range(k):
                idx_views = [idx_s, preds.id_src[:, idx_k]]
                pred_scales[:, idx_k], pred_cosSin[:, idx_k] = (
                    self._model.ist_net.inference(
                        src_feat=src_ist[idx_views],
                        tar_feat=tar_ist,
                        src_pts=preds.src_pts[:, idx_k],
                        tar_pts=preds.tar_pts[:, idx_k],
                    )
                )

            preds.register_tensor("relScale", pred_scales)
            preds.register_tensor("relInplane", pred_cosSin)

            # RANSAC + 6-DoF recovery
            preds = pose_recovery.forward_ransac(predictions=preds)
            score = torch.sum(preds.ransac_scores, dim=2) / num_patches
            preds.register_tensor("scores", score)

            pred_poses = pose_recovery.forward_recovery(
                tar_label=tar_label,
                tar_K=batch.tar_K,
                tar_M=batch.tar_M,
                pred_src_views=preds.id_src,
                pred_M=preds.M.clone(),
            )

        best_idx = score[0].argmax().item()
        pose_matrix = pred_poses[0, best_idx].cpu().numpy()   # [4,4]
        best_score = float(score[0, best_idx].item())

        return pose_matrix, best_score

    def postprocess(self, raw_output, **kwargs) -> SkillResult:
        pose_matrix, score = raw_output
        R = pose_matrix[:3, :3]
        t = pose_matrix[:3, 3]
        return SkillResult(
            skill_name="gigapose",
            success=True,
            data={
                "pose_matrix": pose_matrix,
                "rotation":    R,
                "translation": t,
                "score":       score,
            },
            metadata={
                "n_templates":  self.cfg.n_templates,
                "template_dir": self.cfg.template_dir,
            },
        )

    # ------------------------------------------------------------------
    # Template loading (called once per object, cached)
    # ------------------------------------------------------------------

    def _load_templates(
        self, template_dir: str, obj_id: int, cache_key: str
    ) -> None:
        """Load pre-rendered templates and pre-compute AE/IST features."""
        import torch
        from src.dataloader.template import TemplateSet
        from src.models.poses import ObjectPoseRecovery
        from src.utils.batch import BatchedData
        import src.megapose.utils.tensor_collection as tc
        import pandas as pd

        device = self.cfg.device

        # TemplateSet maps a BOP-format template directory to a dataset.
        # template_dir should contain per-object subdirs (obj_000001, …)
        # or point directly to a single-object directory.
        template_dataset = TemplateSet(
            template_dir=template_dir,
            obj_id=obj_id,
            device=device,
        )

        names = ["rgb", "mask", "K", "M", "poses"]
        extra = ["ae_features", "ist_features"]
        td = {n: BatchedData(None) for n in names + extra}

        with torch.no_grad():
            for idx in range(len(template_dataset)):
                item = template_dataset[idx]
                templates = item.rgb.to(device)
                td["ae_features"].append(self._model.ae_net(templates))
                td["ist_features"].append(
                    self._model.ist_net.forward_by_chunk(templates)
                )
                for n in names:
                    td[n].append(getattr(item, n).to(device))

        for n in names + ["ae_features", "ist_features"]:
            td[n].stack()
            td[n] = td[n].data

        template_data = tc.PandasTensorCollection(
            infos=pd.DataFrame(), **td
        )
        pose_recovery = ObjectPoseRecovery(
            template_K=td["K"],
            template_Ms=td["M"],
            template_poses=td["poses"],
        )

        self._template_cache[cache_key] = {
            "data": template_data,
            "recovery": pose_recovery,
        }
        self.logger.info(
            f"Templates loaded for obj_id={obj_id} "
            f"from {Path(template_dir).name}"
        )
