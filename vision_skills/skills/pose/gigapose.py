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

    4. Download pre-rendered BOP templates (1.4 GB) **or** render your own::

           # Option A — pre-rendered for LMO / TLESS / YCBV / … (recommended)
           mkdir -p ~/gigaPose_datasets/datasets/tmp
           wget -O ~/gigaPose_datasets/datasets/tmp/templates.zip \\
               https://huggingface.co/datasets/nv-nguyen/gigaPose/resolve/main/templates.zip
           unzip ~/gigaPose_datasets/datasets/tmp/templates.zip \\
               -d ~/gigaPose_datasets/datasets/

           # Option B — render custom objects (requires Panda3D + CAD model)
           python ~/libs/gigapose/src/scripts/render_custom_templates.py

    Usage::

        skill = SkillRegistry.create("gigapose", {
            "checkpoint":    "~/libs/gigapose/pretrained/gigaPose_v1.ckpt",
            # Parent directory that contains {obj_id:06d}/ sub-dirs and
            # object_poses/{obj_id:06d}.npy files (BOP template format).
            "template_dir":  "~/gigaPose_datasets/datasets/templates/lmo",
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

        image, mask, K, bbox, obj_id, template_dir = processed
        device = self.cfg.device
        h_img, w_img = image.shape[:2]

        # 1. Load / cache template features
        cache_key = f"{template_dir}::{obj_id}"
        if cache_key not in self._template_cache:
            self._load_templates(template_dir, obj_id, cache_key)

        # 2. Crop query image to bbox using CropResizePad (isotropic scale +
        #    padding) — must match the template preprocessing exactly, because
        #    forward_recovery asserts M[0,0] == M[1,1] (no anisotropic scaling).
        from src.utils.crop import CropResizePad
        import torchvision.transforms as T

        img_tensor = to_tensor(image).to(device)            # [3, H, W]  float
        msk_tensor = torch.from_numpy(
            (mask > 0).astype(np.float32)
        ).unsqueeze(0).to(device)                           # [1, H, W]

        bbox_t = torch.tensor(
            [bbox], dtype=torch.long, device=device         # [1, 4]  xyxy int
        )
        crop = CropResizePad(target_size=224)

        cropped_rgb  = crop(bbox_t, img_tensor.unsqueeze(0))        # images [1,3,224,224], M [1,3,3]
        rgba_for_msk = torch.cat([img_tensor, msk_tensor], dim=0)   # [4, H, W]
        cropped_msk  = crop(bbox_t, rgba_for_msk.unsqueeze(0))      # images [1,4,224,224]

        tar_img  = cropped_rgb["images"]                             # [1, 3, 224, 224]
        tar_mask = cropped_msk["images"][:, 3]                       # [1, 224, 224]
        tar_M    = cropped_rgb["M"]                                  # [1, 3, 3]

        # Apply ImageNet normalisation to the query (same as templates)
        normalize = T.Normalize(
            mean=[0.48145466, 0.4578275,  0.40821073],
            std= [0.26862954, 0.26130258, 0.27577711],
        )
        tar_img = torch.stack([normalize(tar_img[i]) for i in range(len(tar_img))])

        tar_K = torch.from_numpy(K).unsqueeze(0).to(device)         # [1, 3, 3]

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
        """Load pre-rendered templates and pre-compute AE/IST features.

        template_dir is the **parent** directory that contains:
          - {obj_id:06d}/000000.png … 000161.png   (RGBA renders)
          - {obj_id:06d}/000000_depth.png …         (depth renders)
          - object_poses/{obj_id:06d}.npy            (162 SE(3) poses)

        This structure is produced by the gigapose render scripts and matches
        the layout inside the official templates.zip download.
        """
        import torch
        import pandas as pd
        import torchvision.transforms as T
        from src.custom_megapose.template_dataset import TemplateDataset
        from src.models.poses import ObjectPoseRecovery
        from src.utils.crop import CropResizePad
        import src.megapose.utils.tensor_collection as tc

        device = self.cfg.device
        n_templates = self.cfg.n_templates

        # Minimal config that mirrors configs/data/bop.yaml template_config.
        class _TCfg:
            dir = template_dir
            num_templates = n_templates
            pose_name = "object_poses/OBJECT_ID.npy"
            scale_factor = 1.0   # BOP convention (metres)

        dataset = TemplateDataset.from_config([{"obj_id": obj_id}], _TCfg())
        tdata = dataset.get_object_templates(str(obj_id))
        raw, poses = tdata.read_test_mode()
        # raw["rgba"]  [N, 4, H, W] float  (RGBA)
        # raw["box"]   [N, 4]       int    (xyxy bounding boxes in the render)
        # poses        [N, 4, 4]           (SE3 camera←object)

        # Crop-resize RGBA to 224×224 (same as the TemplateSet dataloader).
        crop = CropResizePad(target_size=224)
        images = raw["rgba"].to(device)   # [N, 4, H, W]
        boxes = raw["box"].to(device)     # [N, 4]
        cropped = crop(boxes, images)
        rgb  = cropped["images"][:, :3]   # [N, 3, 224, 224]
        mask = cropped["images"][:, 3]    # [N, 224, 224]
        M    = cropped["M"]               # [N, 3, 3]

        # ImageNet / CLIP normalisation (same as DINOv2 preprocess).
        normalize = T.Normalize(
            mean=[0.48145466, 0.4578275,  0.40821073],
            std= [0.26862954, 0.26130258, 0.27577711],
        )
        rgb = torch.stack([normalize(rgb[i]) for i in range(len(rgb))])

        poses = poses.to(device)          # [N, 4, 4]

        # Fixed Panda3D renderer intrinsics — same hardcoded value as
        # TemplateDataset.K and call_panda3d.py.
        K_np = np.array(
            [572.4114, 0.0, 320.0, 0.0, 573.57043, 240.0, 0.0, 0.0, 1.0],
            dtype=np.float32,
        ).reshape(3, 3)
        K = torch.from_numpy(K_np).to(device)

        # Pre-compute AE and IST features for all template views.
        with torch.no_grad():
            ae_feats  = self._model.ae_net(rgb)                      # [N, D]
            ist_feats = self._model.ist_net.forward_by_chunk(rgb)    # [N, …]

        # Build PandasTensorCollection with a leading object dimension of 1.
        # forward_recovery indexes as template_data[tar_label - 1], so
        # shapes must be [N_objects=1, N_templates, …].
        template_ptc = tc.PandasTensorCollection(
            infos=pd.DataFrame(),
            ae_features=ae_feats.unsqueeze(0),    # [1, N, D]
            ist_features=ist_feats.unsqueeze(0),  # [1, N, …]
            rgb=rgb.unsqueeze(0),                 # [1, N, 3, 224, 224]
            mask=mask.unsqueeze(0),               # [1, N, 224, 224]
            M=M.unsqueeze(0),                     # [1, N, 3, 3]
            K=K.unsqueeze(0),                     # [1, 3, 3]
            poses=poses.unsqueeze(0),             # [1, N, 4, 4]
        )
        pose_recovery = ObjectPoseRecovery(
            template_K=K.unsqueeze(0),            # [1, 3, 3]
            template_Ms=M.unsqueeze(0),           # [1, N, 3, 3]
            template_poses=poses.unsqueeze(0),    # [1, N, 4, 4]
        )

        self._template_cache[cache_key] = {
            "data": template_ptc,
            "recovery": pose_recovery,
        }
        self.logger.info(
            f"Templates loaded for obj_id={obj_id} ({n_templates} views) "
            f"from {Path(template_dir).name}"
        )
