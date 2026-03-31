#!/usr/bin/env python3
"""
Unit tests for VerificationService.

Tests WITHOUT real ML models or infrastructure:
- validate: image/mask/dino_result/superpoint_result/known_tree_id validation
- verify: happy path (hierarchical), not-hierarchical error path, exception returns ERROR dict
- _fine_grained_matching: no lightglue fallback, missing minio_key handling
- get_feature_info: delegates to milvus/minio repos
"""

import asyncio
from dataclasses import dataclass
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.service.verificationService import MatchCandidate, VerificationService


# ---------------------------------------------------------------------------
# Minimal stand-ins for heavy ML results
# ---------------------------------------------------------------------------


@dataclass
class _FakeDinoResult:
    global_descriptor: np.ndarray
    model_name: str = "dino_fake"


@dataclass
class _FakeSuperPointResult:
    keypoints: np.ndarray
    descriptors: np.ndarray
    scores: Optional[np.ndarray] = None


def _rng_vec(n: int = 128, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.random(n).astype(np.float32)
    return v / np.linalg.norm(v)


def _good_dino():
    return _FakeDinoResult(global_descriptor=_rng_vec(128))


def _good_superpoint(n_kp: int = 20, desc_dim: int = 256):
    rng = np.random.default_rng(1)
    return _FakeSuperPointResult(
        keypoints=rng.random((n_kp, 2)).astype(np.float32) * 640,
        descriptors=rng.random((n_kp, desc_dim)).astype(np.float32),
        scores=rng.random(n_kp).astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Helper: build a VerificationService with all heavy deps mocked out
# ---------------------------------------------------------------------------


def _make_service(
    use_hierarchical: bool = True,
    hierarchical_service=None,
    verification_service=None,
    lightglue=None,
) -> VerificationService:
    svc = VerificationService.__new__(VerificationService)

    # Core config
    svc.top_k = 10
    svc.coarse_threshold = 0.6
    svc.inlier_threshold = 15
    svc.use_hierarchical = use_hierarchical
    svc.verbose = False

    # External processors / repos — all mocked
    svc.preprocessor = MagicMock()
    svc.dino_processor = MagicMock()
    svc.superpoint_processor = MagicMock()
    svc.lightglue_processor = lightglue  # None or Mock
    svc.milvus_repo = MagicMock()
    svc.minio_repo = MagicMock()
    svc.texture_processor = None

    # Matching strategy
    svc.matching_strategy = MagicMock()

    # Hierarchical components
    svc.hierarchical_service = hierarchical_service or MagicMock()
    svc.verification_service = verification_service or MagicMock()

    return svc


# ===========================================================================
# validate()
# ===========================================================================


class TestValidate:
    """Tests for VerificationService.validate()."""

    def setup_method(self):
        self.svc = _make_service()

    # --- image ---

    def test_valid_image_passes(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        ok, msg = self.svc.validate(image=img)
        assert ok is True
        assert msg is None

    def test_image_not_ndarray_fails(self):
        ok, msg = self.svc.validate(image="not_an_array")
        assert ok is False
        assert "numpy array" in msg.lower() or "invalid image" in msg.lower()

    def test_empty_image_fails(self):
        ok, msg = self.svc.validate(image=np.array([]))
        assert ok is False
        assert "empty" in msg.lower()

    def test_invalid_image_shape_fails(self):
        bad = np.zeros((5,), dtype=np.uint8)  # 1-D
        ok, msg = self.svc.validate(image=bad)
        assert ok is False

    def test_2d_grayscale_image_passes(self):
        gray = np.zeros((100, 100), dtype=np.uint8)
        ok, msg = self.svc.validate(image=gray)
        assert ok is True

    # --- mask ---

    def test_valid_mask_passes(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        ok, msg = self.svc.validate(image=img, mask=mask)
        assert ok is True

    def test_mask_shape_mismatch_fails(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.zeros((50, 50), dtype=np.uint8)
        ok, msg = self.svc.validate(image=img, mask=mask)
        assert ok is False
        assert "shape" in msg.lower() or "match" in msg.lower()

    def test_mask_not_ndarray_fails(self):
        ok, msg = self.svc.validate(mask="not_a_mask")
        assert ok is False

    def test_empty_mask_fails(self):
        ok, msg = self.svc.validate(mask=np.array([]))
        assert ok is False
        assert "empty" in msg.lower()

    # --- dino_result ---

    def test_valid_dino_result_passes(self):
        ok, msg = self.svc.validate(dino_result=_good_dino())
        assert ok is True

    def test_dino_missing_descriptor_fails(self):
        bad = MagicMock(spec=[])  # no attributes
        ok, msg = self.svc.validate(dino_result=bad)
        assert ok is False

    def test_dino_empty_descriptor_fails(self):
        bad = _FakeDinoResult(global_descriptor=np.array([]))
        ok, msg = self.svc.validate(dino_result=bad)
        assert ok is False

    def test_dino_nan_descriptor_fails(self):
        bad = _FakeDinoResult(global_descriptor=np.array([np.nan, 1.0]))
        ok, msg = self.svc.validate(dino_result=bad)
        assert ok is False
        assert "nan" in msg.lower() or "dino" in msg.lower()

    def test_dino_inf_descriptor_fails(self):
        bad = _FakeDinoResult(global_descriptor=np.array([np.inf, 1.0]))
        ok, msg = self.svc.validate(dino_result=bad)
        assert ok is False

    # --- superpoint_result ---

    def test_valid_superpoint_result_passes(self):
        ok, msg = self.svc.validate(superpoint_result=_good_superpoint())
        assert ok is True

    def test_superpoint_missing_keypoints_attr_fails(self):
        bad = MagicMock(spec=[])  # missing attributes
        ok, msg = self.svc.validate(superpoint_result=bad)
        assert ok is False

    def test_superpoint_empty_descriptors_fails(self):
        sp = _FakeSuperPointResult(
            keypoints=np.zeros((10, 2), dtype=np.float32),
            descriptors=np.array([], dtype=np.float32),
        )
        ok, msg = self.svc.validate(superpoint_result=sp)
        assert ok is False

    def test_superpoint_invalid_keypoint_shape_fails(self):
        sp = _FakeSuperPointResult(
            keypoints=np.zeros((10, 3), dtype=np.float32),  # should be Nx2
            descriptors=np.zeros((10, 256), dtype=np.float32),
        )
        ok, msg = self.svc.validate(superpoint_result=sp)
        assert ok is False

    def test_superpoint_nan_keypoints_fails(self):
        kp = np.zeros((5, 2), dtype=np.float32)
        kp[0, 0] = np.nan
        sp = _FakeSuperPointResult(
            keypoints=kp,
            descriptors=np.zeros((5, 256), dtype=np.float32),
        )
        ok, msg = self.svc.validate(superpoint_result=sp)
        assert ok is False

    def test_superpoint_2d_descriptors_passes(self):
        sp = _good_superpoint()
        ok, msg = self.svc.validate(superpoint_result=sp)
        assert ok is True

    def test_superpoint_1d_descriptors_fails(self):
        sp = _FakeSuperPointResult(
            keypoints=np.zeros((5, 2), dtype=np.float32),
            descriptors=np.zeros(256, dtype=np.float32),  # 1-D
        )
        ok, msg = self.svc.validate(superpoint_result=sp)
        assert ok is False

    # --- known_tree_id ---

    def test_valid_tree_id_passes(self):
        ok, msg = self.svc.validate(known_tree_id="TREE_001")
        assert ok is True

    def test_non_string_tree_id_fails(self):
        ok, msg = self.svc.validate(known_tree_id=123)
        assert ok is False

    def test_empty_tree_id_fails(self):
        ok, msg = self.svc.validate(known_tree_id="   ")
        assert ok is False

    def test_no_args_passes(self):
        ok, msg = self.svc.validate()
        assert ok is True
        assert msg is None


# ===========================================================================
# verify()
# ===========================================================================


class TestVerify:
    """Tests for VerificationService.verify() (async)."""

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    # --- not hierarchical ---

    def test_raises_value_error_when_hierarchical_disabled(self):
        svc = _make_service(use_hierarchical=False)
        img = np.zeros((100, 100, 3), dtype=np.uint8)

        result = self._run(svc.verify(image=img, mask=None))

        assert result["decision"] == "ERROR"
        assert (
            "hierarchical" in result["reason"].lower() or "not enabled" in result["reason"].lower()
        )

    # --- validation failure propagated as ERROR dict ---

    def test_invalid_image_returns_error_dict(self):
        svc = _make_service()
        result = self._run(svc.verify(image="bad", mask=None))

        assert result["decision"] == "ERROR"
        assert "best_match" in result
        assert result["best_match"] is None

    # --- happy path ---

    def test_verify_calls_hierarchical_service_match(self):
        svc = _make_service()

        # Stub _prepare_verification
        dino = _good_dino()
        sp = _good_superpoint()
        svc._prepare_verification = MagicMock(return_value=(dino, sp, None))

        expected_result = {
            "decision": "MATCH",
            "confidence": 0.9,
            "best_match": {"tree_id": "TREE_1", "minio_key": "features/x.npz.gz"},
        }
        svc.hierarchical_service.match_async = AsyncMock(return_value=expected_result)

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = self._run(svc.verify(image=img, mask=None))

        assert result["decision"] == "MATCH"
        svc.hierarchical_service.match_async.assert_called_once()

    def test_verify_passes_geo_filter(self):
        svc = _make_service()
        svc._prepare_verification = MagicMock(return_value=(_good_dino(), _good_superpoint(), None))
        svc.hierarchical_service.match_async = AsyncMock(
            return_value={"decision": "NO_MATCH", "best_match": None, "all_matches": []}
        )

        geo = {"lat_min": 10.0, "lat_max": 20.0, "lon_min": 100.0, "lon_max": 110.0}
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        self._run(svc.verify(image=img, mask=None, geo_filter=geo))

        call_kwargs = svc.hierarchical_service.match_async.call_args[1]
        assert call_kwargs.get("geo_filter") == geo

    def test_verify_passes_angle_filter(self):
        svc = _make_service()
        svc._prepare_verification = MagicMock(return_value=(_good_dino(), _good_superpoint(), None))
        svc.hierarchical_service.match_async = AsyncMock(
            return_value={"decision": "NO_MATCH", "best_match": None, "all_matches": []}
        )

        angle = {"hor_angle_min": 30.0, "hor_angle_max": 90.0}
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        self._run(svc.verify(image=img, mask=None, angle_filter=angle))

        call_kwargs = svc.hierarchical_service.match_async.call_args[1]
        assert call_kwargs.get("angle_filter") == angle

    def test_hierarchical_returns_none_causes_error_dict(self):
        svc = _make_service()
        svc._prepare_verification = MagicMock(return_value=(_good_dino(), _good_superpoint(), None))
        svc.hierarchical_service.match_async = AsyncMock(return_value=None)

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = self._run(svc.verify(image=img, mask=None))

        assert result["decision"] == "ERROR"

    def test_exception_in_prepare_verification_returns_error_dict(self):
        svc = _make_service()
        svc._prepare_verification = MagicMock(side_effect=RuntimeError("GPU OOM"))

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = self._run(svc.verify(image=img, mask=None))

        assert result["decision"] == "ERROR"
        assert "GPU OOM" in result["reason"]


# ===========================================================================
# _fine_grained_matching()
# ===========================================================================


@pytest.mark.skip(reason="_fine_grained_matching removed from VerificationService")
class TestFineGrainedMatching:
    """Tests for removed VerificationService._fine_grained_matching()."""

    def _make_candidate(
        self, image_id: str, tree_id: str, score: float, minio_key: str = None
    ) -> MatchCandidate:
        c = MatchCandidate(image_id=image_id, tree_id=tree_id, similarity_score=score, rank=0)
        c.metadata = {"minio_key": minio_key} if minio_key else {}
        return c

    def test_no_lightglue_returns_coarse_scores_only(self):
        svc = _make_service(lightglue=None)
        candidates = [self._make_candidate("img_1", "tree_A", 0.9, "features/img_1.npz.gz")]
        query_local = {}

        results = svc._fine_grained_matching(query_local, candidates)

        assert len(results) == 1
        assert results[0]["n_inliers"] == 0

    def test_missing_minio_key_skips_candidate(self):
        svc = _make_service(lightglue=MagicMock())
        candidates = [self._make_candidate("img_1", "tree_A", 0.9, minio_key=None)]
        query_local = {}

        results = svc._fine_grained_matching(query_local, candidates)

        # Candidate skipped because no minio_key
        assert results == []

    def test_lightglue_result_extracted(self):
        lg = MagicMock()
        match_result = MagicMock(n_inliers=25, match_score=0.85, confidence=0.9)
        lg.match = MagicMock(return_value=match_result)

        svc = _make_service(lightglue=lg)
        svc.minio_repo.load_features_by_key = MagicMock(
            return_value={"keypoints": np.zeros((10, 2))}
        )

        candidates = [self._make_candidate("img_1", "tree_A", 0.9, "features/img_1.npz.gz")]
        query_local = {"keypoints": np.zeros((10, 2))}

        results = svc._fine_grained_matching(query_local, candidates)

        assert len(results) == 1
        assert results[0]["n_inliers"] == 25
        assert results[0]["match_score"] == pytest.approx(0.85)
        assert results[0]["confidence"] == pytest.approx(0.9)

    def test_minio_exception_skips_candidate(self):
        lg = MagicMock()
        svc = _make_service(lightglue=lg)
        svc.minio_repo.load_features_by_key = MagicMock(side_effect=Exception("Storage error"))

        candidates = [self._make_candidate("img_1", "tree_A", 0.9, "features/img_1.npz.gz")]
        results = svc._fine_grained_matching({}, candidates)

        assert results == []


# ===========================================================================
# _prepare_verification()
# ===========================================================================


class TestPrepareVerification:
    """Tests for VerificationService._prepare_verification() - mask handling."""

    def _make_fake_dino(self):
        return _FakeDinoResult(global_descriptor=_rng_vec(128))

    def _make_fake_superpoint(self):
        return _good_superpoint()

    def _make_service_full(self):
        """Create service with mocked processors."""
        svc = _make_service()

        # Mock preprocessor methods
        svc.preprocessor.apply_mask = MagicMock(
            return_value=np.zeros((100, 100, 3), dtype=np.uint8)
        )
        svc.preprocessor.get_bounding_box = MagicMock(return_value=(0, 0, 100, 100))
        svc.preprocessor.crop_to_bounding_box = MagicMock(
            return_value=(
                np.zeros((50, 50, 3), dtype=np.uint8),
                np.zeros((50, 50), dtype=np.uint8),
                (0, 0, 50, 50),
            )
        )
        svc.preprocessor.prepare_for_dino = MagicMock(
            return_value=np.zeros((3, 224, 224), dtype=np.uint8)
        )
        svc.preprocessor.prepare_for_superpoint = MagicMock(
            return_value=(np.zeros((640, 640), dtype=np.uint8), None)
        )
        svc.preprocessor.to_grayscale = MagicMock(return_value=np.zeros((640, 640), dtype=np.uint8))
        svc.preprocessor.segment_with_sam3 = MagicMock(
            return_value=(None, np.zeros((100, 100), dtype=np.uint8))
        )

        # Mock processors
        svc.dino_processor.extract = MagicMock(return_value=self._make_fake_dino())
        svc.superpoint_processor.extract = MagicMock(return_value=self._make_fake_superpoint())
        svc.texture_processor = None

        return svc

    def test_uses_provided_grayscale_mask(self):
        """When mask is provided as grayscale, use it directly (no SAM3)."""
        svc = self._make_service_full()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.ones((100, 100), dtype=np.uint8) * 255

        dino, sp, texture = svc._prepare_verification(img, mask=mask)

        # SAM3 should NOT be called
        svc.preprocessor.segment_with_sam3.assert_not_called()
        # apply_mask should be called with the provided mask
        svc.preprocessor.apply_mask.assert_called_once_with(img, mask)
        assert dino is not None
        assert sp is not None

    def test_uses_provided_bgr_mask_converted_to_gray(self):
        """When mask is provided as BGR, convert to grayscale first."""
        svc = self._make_service_full()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        mask_bgr = np.ones((100, 100, 3), dtype=np.uint8) * 255

        dino, sp, texture = svc._prepare_verification(img, mask=mask_bgr)

        # SAM3 should NOT be called
        svc.preprocessor.segment_with_sam3.assert_not_called()
        # apply_mask should be called (mask converted to grayscale)
        assert svc.preprocessor.apply_mask.called
        assert dino is not None

    def test_falls_back_to_sam3_when_mask_none(self):
        """When mask is None, run SAM3 to generate mask."""
        svc = self._make_service_full()
        img = np.zeros((100, 100, 3), dtype=np.uint8)

        dino, sp, texture = svc._prepare_verification(img, mask=None)

        # SAM3 SHOULD be called
        svc.preprocessor.segment_with_sam3.assert_called_once_with(img)
        assert dino is not None
        assert sp is not None

    def test_returns_correct_tuple_types(self):
        """Verify return types match expected signature."""
        svc = self._make_service_full()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.ones((100, 100), dtype=np.uint8) * 255

        result = svc._prepare_verification(img, mask=mask)

        assert isinstance(result, tuple)
        assert len(result) == 3
        # First element is DINO result
        assert hasattr(result[0], "global_descriptor")
        # Second is SuperPoint result
        assert hasattr(result[1], "keypoints")
        assert hasattr(result[1], "descriptors")
        # Third is texture (None in this case)
        assert result[2] is None

    def test_same_pipeline_as_ingestion(self):
        """Verify the pipeline mirrors IngestionService exactly."""
        svc = self._make_service_full()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.ones((100, 100), dtype=np.uint8) * 255

        svc._prepare_verification(img, mask=mask)

        # Verify all steps that mirror ingestion
        svc.preprocessor.apply_mask.assert_called()
        svc.preprocessor.get_bounding_box.assert_called()
        svc.preprocessor.crop_to_bounding_box.assert_called()
        svc.preprocessor.prepare_for_dino.assert_called()
        svc.preprocessor.prepare_for_superpoint.assert_called()
        svc.preprocessor.to_grayscale.assert_called()
        svc.dino_processor.extract.assert_called()
        svc.superpoint_processor.extract.assert_called()


# ===========================================================================
# get_feature_info()
# ===========================================================================


@pytest.mark.skip(reason="get_feature_info removed from VerificationService")
@pytest.mark.skip(reason="get_feature_info removed from VerificationService")
class TestGetFeatureInfo:
    def test_returns_milvus_and_minio_info(self):
        svc = _make_service()
        svc.milvus_repo.get_collection_info = MagicMock(return_value={"entity_count": 42})
        svc.minio_repo.get_feature_count = MagicMock(
            return_value={"total_features": 15, "features_by_tree": {"T1": 5}}
        )

        info = svc.get_feature_info()

        assert info["total_entities"] == 42
        assert info["total_features"] == 15
        assert "T1" in info["features_by_tree"]

    def test_exception_returns_error_dict(self):
        svc = _make_service()
        svc.milvus_repo.get_collection_info = MagicMock(side_effect=Exception("DB down"))

        info = svc.get_feature_info()

        assert "error" in info
        assert info["total_entities"] == 0
