import numpy as np
import torch
from collections import Counter
from typing import Dict, List, Optional, Tuple
from PIL import Image
from ...core.registry import register_skill
from ...core.base_skill import SkillResult
from ...config.skill_configs import ClassificationConfig
from .base_classification import BaseClassificationSkill


@register_skill("dinov2")
class DINOv2Skill(BaseClassificationSkill):
    """
    DINOv2 few-shot classification + anomaly detection.

    Two scoring modes via ClassificationConfig.patch_mode:

    CLS mode (patch_mode=False, default):
        Single CLS-token embedding per image → k-NN voting and image-level
        anomaly score.  Fast; good for large structural differences.

    Patch mode (patch_mode=True):
        All patch-token embeddings from gallery images are stored in FAISS.
        At inference every patch is matched to its k nearest gallery patches;
        per-patch anomaly scores are aggregated to an image-level score and a
        spatial anomaly map.  Much more sensitive to local surface defects.

    Workflow (patch mode):
        dino = SkillRegistry.create("dinov2", {"patch_mode": True, ...})
        dino.build_gallery({"good": [img1, img2, ...]})
        result = dino(test_image)
        # result.data["anomaly_map"]  [h,w] float [0,1], higher=more anomalous
        # result.data["image_score"]  scalar (lower = more anomalous)
        # result.data["is_anomaly"]   bool
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.cfg = ClassificationConfig(**config)
        self.label_map: List[str] = []
        self.index = None

    def load(self) -> None:
        from transformers import AutoImageProcessor, AutoModel

        dtype = (
            torch.float16 if self.cfg.device == "cuda" else torch.float32
        )
        self.processor = AutoImageProcessor.from_pretrained(
            self.cfg.model_name
        )
        self.model = AutoModel.from_pretrained(
            self.cfg.model_name, torch_dtype=dtype
        ).to(self.cfg.device)
        self.model.eval()
        self.logger.info(f"DINOv2 loaded: {self.cfg.model_name}")

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _processor_kwargs(self) -> dict:
        """Extra kwargs for the image processor when image_size != 224."""
        if self.cfg.image_size == 224:
            return {}
        sz = {"height": self.cfg.image_size, "width": self.cfg.image_size}
        return {"size": sz, "crop_size": sz}

    def _embed(self, image: np.ndarray) -> np.ndarray:
        """L2-normalised CLS-token embedding. Returns [1, D]."""
        pil = Image.fromarray(image)
        inputs = self.processor(
            images=pil, return_tensors="pt", **self._processor_kwargs()
        )
        inputs = {k: v.to(self.cfg.device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = self.model(**inputs).last_hidden_state[:, 0]
        emb = feats.float().cpu().numpy()
        norm = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
        return emb / norm  # [1, D]

    def _embed_patches(self, image: np.ndarray) -> np.ndarray:
        """L2-normalised patch-token embeddings. Returns [n_patches, D]."""
        pil = Image.fromarray(image)
        inputs = self.processor(
            images=pil, return_tensors="pt", **self._processor_kwargs()
        )
        inputs = {k: v.to(self.cfg.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model(**inputs)
        # last_hidden_state: [1, 1+n_patches, D]; index 0 is CLS
        tokens = out.last_hidden_state[0, 1:].float().cpu().numpy()
        norm = np.linalg.norm(tokens, axis=1, keepdims=True) + 1e-8
        return tokens / norm  # [n_patches, D]

    # ------------------------------------------------------------------
    # Gallery
    # ------------------------------------------------------------------

    def build_gallery(
        self, images_by_class: Dict[str, List[np.ndarray]]
    ) -> None:
        """
        Build a FAISS flat-IP index from reference images.

        CLS mode:   one vector per image.
        Patch mode: all patch vectors per image concatenated
                    (50 imgs x 256 patches = 12 800 vecs for dinov2-small).

        Args:
            images_by_class: class_name → list of HxWx3 uint8 RGB images.
        """
        import faiss

        if not self._loaded:
            self.load()
            self._loaded = True

        all_embs: List[np.ndarray] = []
        all_labels: List[str] = []

        for label, imgs in images_by_class.items():
            for img in imgs:
                if self.cfg.patch_mode:
                    patches = self._embed_patches(img)  # [n_patches, D]
                    all_embs.append(patches)
                    all_labels.extend([label] * len(patches))
                else:
                    emb = self._embed(img)              # [1, D]
                    all_embs.append(emb)
                    all_labels.append(label)

        matrix = np.vstack(all_embs).astype(np.float32)
        self.label_map = all_labels

        d = matrix.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(matrix)

        mode = "patch" if self.cfg.patch_mode else "CLS"
        n_imgs = sum(len(v) for v in images_by_class.values())
        self.logger.info(
            f"Gallery built ({mode} mode): {len(matrix)} vectors "
            f"from {n_imgs} images."
        )

    # ------------------------------------------------------------------
    # BaseSkill interface
    # ------------------------------------------------------------------

    def preprocess(self, image: np.ndarray, **kwargs) -> np.ndarray:
        if self.cfg.patch_mode:
            return self._embed_patches(image)  # [n_patches, D]
        return self._embed(image)              # [1, D]

    def infer(
        self,
        embedding: np.ndarray,
        k: Optional[int] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.index is None:
            raise RuntimeError(
                "Gallery not built. Call build_gallery() first."
            )
        k = k or self.cfg.few_shot_k
        sims, idxs = self.index.search(embedding.astype(np.float32), k)
        return sims, idxs  # both [n_queries, k]

    def postprocess(
        self,
        raw_output: Tuple[np.ndarray, np.ndarray],
        threshold: Optional[float] = None,
        **kwargs,
    ) -> SkillResult:
        sims, idxs = raw_output
        threshold = (
            threshold
            if threshold is not None
            else self.cfg.anomaly_threshold
        )
        if self.cfg.patch_mode:
            return self._postprocess_patches(sims, threshold)
        return self._postprocess_cls(sims[0], idxs[0], threshold)

    # ------------------------------------------------------------------
    # Postprocess helpers
    # ------------------------------------------------------------------

    def _postprocess_patches(
        self, sims: np.ndarray, threshold: float
    ) -> SkillResult:
        """
        sims: [n_patches, k] per-patch cosine similarities to gallery.

        Pipeline:
          1. Average over k neighbours → patch_sims [n_patches]
          2. Reshape to (side x side) patch grid
          3. Optional Gaussian smoothing (cfg.smooth_sigma > 0):
             amplifies spatially-coherent anomaly regions (bent edges)
             and suppresses isolated noisy patches
          4. Image-level score: mean of cfg.anomaly_top_frac worst patches
             (lower = more anomalous; mirrors CLS confidence semantics)
        """
        # 1. Per-patch mean similarity over k nearest gallery neighbours
        patch_sims = sims.mean(axis=1)  # [n_patches]

        # 2. Reshape to spatial grid
        n = len(patch_sims)
        side = int(np.sqrt(n))
        sims_2d = patch_sims.reshape(side, side)

        # 3. Optional spatial smoothing
        if self.cfg.smooth_sigma > 0:
            from scipy.ndimage import gaussian_filter
            sims_2d = gaussian_filter(
                sims_2d, sigma=self.cfg.smooth_sigma
            )

        patch_sims_eff = sims_2d.flatten()

        # 4. Image-level score from the worst top_frac patches
        n_top = max(
            1, int(len(patch_sims_eff) * self.cfg.anomaly_top_frac)
        )
        image_score = float(np.sort(patch_sims_eff)[:n_top].mean())
        is_anomaly = image_score < threshold

        # Spatial anomaly map normalised to [0, 1]
        anomaly_raw = (1.0 - sims_2d).astype(np.float32)
        a_min, a_max = anomaly_raw.min(), anomaly_raw.max()
        anomaly_map = (anomaly_raw - a_min) / (a_max - a_min + 1e-8)

        return SkillResult(
            skill_name="dinov2",
            success=True,
            data={
                "anomaly_map":     anomaly_map,   # [h, w] float [0,1]
                "anomaly_map_raw": anomaly_raw,   # [h, w] unnormalised
                "image_score":     image_score,   # lower = more anomalous
                "confidence":      image_score,   # CLS-mode compat alias
                "is_anomaly":      is_anomaly,
                "patch_sims":      patch_sims,    # [n_patches] original
            },
            metadata={
                "threshold":    threshold,
                "patch_grid":   (side, side),
                "top_frac":     self.cfg.anomaly_top_frac,
                "smooth_sigma": self.cfg.smooth_sigma,
            },
        )

    def _postprocess_cls(
        self,
        sims: np.ndarray,
        idxs: np.ndarray,
        threshold: float,
    ) -> SkillResult:
        """k-NN voting + anomaly flag from CLS-token similarities."""
        labels = [self.label_map[i] for i in idxs]
        vote = Counter(labels)
        predicted_class, _ = vote.most_common(1)[0]
        confidence = float(np.mean(sims))
        is_anomaly = confidence < threshold

        return SkillResult(
            skill_name="dinov2",
            success=True,
            data={
                "predicted_class":    (
                    "anomaly" if is_anomaly else predicted_class
                ),
                "confidence":         confidence,
                "is_anomaly":         is_anomaly,
                "top_k_labels":       labels,
                "top_k_similarities": sims.tolist(),
                "votes":              dict(vote),
            },
            metadata={"threshold": threshold, "k": len(idxs)},
        )
