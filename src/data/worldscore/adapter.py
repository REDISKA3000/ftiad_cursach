from __future__ import annotations

import importlib
from copy import deepcopy
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from pprint import pprint
from typing import Any, Iterable, Iterator

from PIL import Image


@dataclass
class WorldScoreSample:
    sample_id: str
    split_name: str
    raw_index: int
    image_path: str
    positive_prompt: str
    negative_prompt: str
    world_spec: dict
    metadata: dict = field(default_factory=dict)


def _build_world_spec_compat(
    image_path: str,
    next_scene_prompt: str,
    camera_prompt: str = "static camera",
    target_objects=None,
    main_subject: str | None = None,
    secondary_subjects=None,
    expected_action: str | None = None,
    scene_context: str | None = None,
    style_tags=None,
    consistency_constraints=None,
    chunk_idx: int = 0,
    previous_chunk_last_frame: str | None = None,
    model_name: str = "custom_i2v_model",
    model_type: str = "videogen",
    generate_type: str = "i2v",
    frames_per_chunk: int = 40,
    fps: int = 8,
    resolution=(320, 512),
    num_chunks: int = 3,
    camera_trajectory_gt=None,
    camera_transform_hint=None,
    motion_mask_hint=None,
    expected_scene_transition: str | None = None,
    chunk_prompt_schedule=None,
    entity_region_hints=None,
    affected_entities=None,
    unaffected_entities=None,
    expected_motion=None,
    event_sequence=None,
    revisit_pairs=None,
):
    if target_objects is None:
        target_objects = []
    if secondary_subjects is None:
        secondary_subjects = []
    if style_tags is None:
        style_tags = []
    if consistency_constraints is None:
        consistency_constraints = [
            "keep the same subject identity",
            "keep the same scene",
            "avoid sudden scene changes",
            "avoid extra objects",
        ]
    if chunk_prompt_schedule is None:
        chunk_prompt_schedule = {}
    if entity_region_hints is None:
        entity_region_hints = {}

    resolution = list(resolution)
    layout_spec = {
        "camera_prompt": camera_prompt,
        "camera_instruction_text": camera_prompt,
        "camera_trajectory_gt": camera_trajectory_gt,
        "camera_transform_hint": camera_transform_hint,
    }

    return {
        "chunk_idx": chunk_idx,
        "num_chunks": num_chunks,
        "current_scene_image": str(image_path),
        "previous_chunk_last_frame": previous_chunk_last_frame,
        "model_name": model_name,
        "model_type": model_type,
        "generate_type": generate_type,
        "frames_per_chunk": frames_per_chunk,
        "fps": fps,
        "resolution": resolution,
        "next_scene_prompt": next_scene_prompt,
        "expected_scene_transition": expected_scene_transition,
        "camera_prompt": camera_prompt,
        "camera_instruction_text": camera_prompt,
        "layout_spec": layout_spec,
        "camera_trajectory_gt": camera_trajectory_gt,
        "camera_transform_hint": camera_transform_hint,
        "motion_mask_hint": motion_mask_hint,
        "entity_region_hints": deepcopy(entity_region_hints),
        "main_subject": main_subject,
        "secondary_subjects": list(secondary_subjects),
        "target_objects": list(target_objects),
        "expected_action": expected_action,
        "scene_context": scene_context,
        "style_tags": list(style_tags),
        "consistency_constraints": list(consistency_constraints),
        "chunk_prompt_schedule": chunk_prompt_schedule,
        "affected_entities": deepcopy(affected_entities),
        "unaffected_entities": deepcopy(unaffected_entities),
        "expected_motion": deepcopy(expected_motion),
        "event_sequence": deepcopy(event_sequence),
        "revisit_pairs": deepcopy(revisit_pairs),
    }


class WorldScoreDatasetAdapter:
    """Normalize Howieeeee/WorldScore rows into pipeline-ready samples."""

    CAMERA_PATH_MAP = {
        "fixed": "fixed camera",
        "push_in": "camera push in",
        "pull_out": "camera pull out",
        "move_left": "camera move left",
        "move_right": "camera move right",
        "orbit_left": "camera orbit left",
        "orbit_right": "camera orbit right",
    }

    def __init__(
        self,
        split_name: str,
        cache_dir: str,
        image_subdir: str = "images",
        dataset_name: str = "Howieeeee/WorldScore",
        negative_prompt: str = "",
        frames_per_chunk: int = 40,
        fps: int = 8,
        resolution: tuple[int, int] = (320, 512),
        num_chunks: int = 1,
        overwrite_images: bool = False,
        dataset=None,
    ):
        self.dataset_name = dataset_name
        self.split_name = split_name
        self.cache_dir = Path(cache_dir)
        self.image_dir = self.cache_dir / image_subdir
        self.negative_prompt = negative_prompt
        self.frames_per_chunk = frames_per_chunk
        self.fps = fps
        self.resolution = resolution
        self.num_chunks = num_chunks
        self.overwrite_images = overwrite_images
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.ds = dataset if dataset is not None else self._load_dataset_split()

    def __len__(self) -> int:
        return len(self.ds)

    def get_raw(self, idx: int) -> dict:
        return dict(self.ds[idx])

    def get_sample(self, idx: int) -> WorldScoreSample:
        row = self.get_raw(idx)
        image_path = self._save_image(idx, row.get("image"))
        positive_prompt = self._build_positive_prompt(row)
        camera_prompt = self._build_camera_prompt(row)
        style_tags = self._build_style_tags(row)
        main_subject, secondary_subjects = self._build_subjects(row)
        scene_context = self._build_scene_context(row)
        expected_action = self._build_expected_action(row)
        target_objects = self._build_target_objects(row)

        derived = {
            "camera_prompt": camera_prompt,
            "style_tags": style_tags,
            "main_subject": main_subject,
            "secondary_subjects": secondary_subjects,
            "target_objects": target_objects,
            "scene_context": scene_context,
            "expected_action": expected_action,
        }

        world_spec = self._build_world_spec(
            row=row,
            image_path=image_path,
            positive_prompt=positive_prompt,
            derived=derived,
        )
        metadata = self._build_metadata(row=row, derived=derived, raw_index=idx)

        return WorldScoreSample(
            sample_id=f"{self.split_name}_{idx:06d}",
            split_name=self.split_name,
            raw_index=idx,
            image_path=image_path,
            positive_prompt=positive_prompt,
            negative_prompt=self.negative_prompt,
            world_spec=world_spec,
            metadata=metadata,
        )

    def iter_samples(self, indices=None) -> Iterator[WorldScoreSample]:
        if indices is None:
            indices = range(len(self))
        for idx in indices:
            yield self.get_sample(int(idx))

    def preview(self, idx: int):
        sample = self.get_sample(idx)
        image_size = None
        try:
            with Image.open(sample.image_path) as img:
                image_size = img.size
        except Exception:
            image_size = None

        payload = {
            "sample_id": sample.sample_id,
            "split_name": sample.split_name,
            "raw_index": sample.raw_index,
            "image_path": sample.image_path,
            "image_size": image_size,
            "positive_prompt": sample.positive_prompt,
            "negative_prompt": sample.negative_prompt,
            "world_spec_summary": {
                "main_subject": sample.world_spec.get("main_subject"),
                "secondary_subjects": sample.world_spec.get("secondary_subjects"),
                "target_objects": sample.world_spec.get("target_objects"),
                "camera_prompt": sample.world_spec.get("camera_prompt"),
                "expected_action": sample.world_spec.get("expected_action"),
                "scene_context": sample.world_spec.get("scene_context"),
                "style_tags": sample.world_spec.get("style_tags"),
            },
            "metadata_keys": sorted(sample.metadata.keys()),
        }
        pprint(payload)
        return payload

    def _load_dataset_split(self):
        try:
            datasets = importlib.import_module("datasets")
        except Exception as exc:
            raise RuntimeError(
                "datasets library is required to load Howieeeee/WorldScore."
            ) from exc

        # WorldScore config = split_name ('dynamic'/'static'), inner split = 'train'
        ds = datasets.load_dataset(self.dataset_name, self.split_name)
        return ds["train"] if isinstance(ds, dict) and "train" in ds else ds

    def _save_image(self, idx: int, image_obj) -> str:
        image_path = self.image_dir / f"{self.split_name}_{idx:06d}.png"
        if image_path.exists() and not self.overwrite_images:
            return str(image_path)

        image = self._coerce_image(image_obj)
        image.save(image_path)
        return str(image_path)

    def _coerce_image(self, image_obj) -> Image.Image:
        if isinstance(image_obj, Image.Image):
            return image_obj.convert("RGB")

        if isinstance(image_obj, dict):
            if "path" in image_obj and image_obj["path"]:
                return Image.open(image_obj["path"]).convert("RGB")
            if "bytes" in image_obj and image_obj["bytes"] is not None:
                return Image.open(BytesIO(image_obj["bytes"])).convert("RGB")

        if isinstance(image_obj, str):
            return Image.open(image_obj).convert("RGB")

        raise ValueError(f"Unsupported image object type: {type(image_obj)}")

    def _normalize_list(self, value) -> list[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            out = []
            for item in value:
                if item is None:
                    continue
                text = str(item).strip()
                if text:
                    out.append(text)
            return out
        text = str(value).strip()
        return [text] if text else []

    def _dedupe_keep_order(self, values: Iterable[str]) -> list[str]:
        seen = set()
        out = []
        for value in values:
            token = str(value).strip()
            if not token or token in seen:
                continue
            seen.add(token)
            out.append(token)
        return out

    def _build_positive_prompt(self, row: dict) -> str:
        if self.split_name == "static":
            prompt_list = row.get("prompt_list")
            if isinstance(prompt_list, (list, tuple)) and prompt_list:
                prompt = str(prompt_list[0]).strip()
                if prompt:
                    return prompt
        prompt = row.get("prompt")
        if prompt is not None and str(prompt).strip():
            return str(prompt).strip()
        return ""

    def _build_camera_prompt(self, row: dict) -> str:
        camera_path = row.get("camera_path")
        raw_values = self._normalize_list(camera_path)
        token = raw_values[0] if raw_values else ""
        normalized = self.CAMERA_PATH_MAP.get(token)
        if normalized:
            return normalized
        if token:
            return token.replace("_", " ").strip()
        return "static camera"

    def _build_style_tags(self, row: dict) -> list[str]:
        return self._dedupe_keep_order([
            *self._normalize_list(row.get("visual_style")),
            *self._normalize_list(row.get("style")),
        ])

    def _build_subjects(self, row: dict) -> tuple[str | None, list[str]]:
        if self.split_name == "dynamic":
            objects = self._normalize_list(row.get("objects"))
            if objects:
                return objects[0], objects[1:]
            return None, []

        content_list = self._normalize_list(row.get("content_list"))
        if content_list:
            return content_list[0], content_list[1:]
        return None, []

    def _build_scene_context(self, row: dict) -> str | None:
        if self.split_name != "static":
            return None
        parts = [
            *self._normalize_list(row.get("scene_type")),
            *self._normalize_list(row.get("category")),
        ]
        parts = self._dedupe_keep_order(parts)
        if not parts:
            return None
        return ", ".join(parts)

    def _build_expected_action(self, row: dict) -> str | None:
        if self.split_name != "dynamic":
            return None
        motion_type = row.get("motion_type")
        if motion_type is None:
            return None
        value = str(motion_type).strip()
        return value or None

    def _build_target_objects(self, row: dict) -> list[str]:
        if self.split_name == "dynamic":
            return self._normalize_list(row.get("objects"))
        return self._normalize_list(row.get("content_list"))

    def _build_world_spec(self, row: dict, image_path: str, positive_prompt: str, derived: dict) -> dict:
        return _build_world_spec_compat(
            image_path=image_path,
            next_scene_prompt=positive_prompt,
            camera_prompt=derived["camera_prompt"],
            target_objects=derived["target_objects"],
            main_subject=derived["main_subject"],
            secondary_subjects=derived["secondary_subjects"],
            expected_action=derived["expected_action"],
            scene_context=derived["scene_context"],
            style_tags=derived["style_tags"],
            frames_per_chunk=self.frames_per_chunk,
            fps=self.fps,
            resolution=self.resolution,
            num_chunks=self.num_chunks,
        )

    def _build_metadata(self, row: dict, derived: dict, raw_index: int) -> dict:
        metadata = {
            "dataset_name": self.dataset_name,
            "split_name": self.split_name,
            "raw_index": raw_index,
            "visual_movement": row.get("visual_movement"),
            "visual_style": row.get("visual_style"),
            "motion_type": row.get("motion_type"),
            "style": row.get("style"),
            "camera_path": row.get("camera_path"),
            "objects": row.get("objects"),
            "scene_type": row.get("scene_type"),
            "category": row.get("category"),
            "content_list": row.get("content_list"),
            "prompt": row.get("prompt"),
            "prompt_list": row.get("prompt_list"),
            **derived,
        }
        return metadata
