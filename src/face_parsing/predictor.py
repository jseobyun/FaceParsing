from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF

from src.models import FaceParsingModel
from src.utils import FaceParsingVisualizer

ImageLike = Union[str, Path, Image.Image, np.ndarray, torch.Tensor]


@dataclass
class FaceParsingSingleResult:
    """Structured container holding the prediction for one image."""

    mask: torch.Tensor
    probabilities: torch.Tensor
    original_size: Tuple[int, int]
    mask_resized: Optional[torch.Tensor] = None
    source_path: Optional[str] = None

    def numpy_mask(self) -> np.ndarray:
        """Return the resized mask as a numpy array."""
        target = self.mask_resized if self.mask_resized is not None else self.mask
        return target.detach().cpu().numpy().astype(np.uint8)


class FaceParsingPredictor:
    """Plug-and-play helper that loads checkpoints and runs inference."""

    DEFAULT_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        dinov3_checkpoint_path: Optional[Union[str, Path]] = None,
        *,
        device: Optional[Union[str, torch.device]] = None,
        image_size: Tuple[int, int] = (512, 512),
        num_classes: int = 19,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.image_size = tuple(image_size)
        self.num_classes = num_classes
        self.transform = transform or self._build_default_transform(self.image_size)
        self.model = self._load_model(checkpoint_path, dinov3_checkpoint_path)
        self.visualizer: Optional[FaceParsingVisualizer] = None

    def predict(
        self,
        image: ImageLike,
        *,
        resize_to_original: bool = True,
        output_device: Optional[Union[str, torch.device]] = "cpu",
    ) -> FaceParsingSingleResult:
        """Predict a single sample."""
        return self.predict_batch(
            [image],
            resize_to_original=resize_to_original,
            output_device=output_device,
        )[0]

    def predict_batch(
        self,
        images: Sequence[ImageLike],
        *,
        resize_to_original: bool = True,
        output_device: Optional[Union[str, torch.device]] = "cpu",
    ) -> List[FaceParsingSingleResult]:
        """Predict a batch and return structured results per sample."""
        if not images:
            raise ValueError("`images` must contain at least one element.")

        prepared = [self._prepare_image(sample) for sample in images]
        batch = torch.stack([item["tensor"] for item in prepared]).to(self.device)

        with torch.no_grad():
            outputs = self.model(batch)

        target_device = torch.device(output_device) if output_device else self.device
        probabilities = outputs["probabilities"].to(target_device)
        masks = torch.argmax(probabilities, dim=1)

        resized_masks: List[Optional[torch.Tensor]] = [None] * len(prepared)
        if resize_to_original:
            resized_masks = self._resize_masks(masks, [item["original_size"] for item in prepared])

        results: List[FaceParsingSingleResult] = []
        for idx, meta in enumerate(prepared):
            results.append(
                FaceParsingSingleResult(
                    mask=masks[idx].detach(),
                    probabilities=probabilities[idx].detach(),
                    original_size=meta["original_size"],
                    mask_resized=resized_masks[idx],
                    source_path=meta["source_path"],
                )
            )

        return results

    def predict_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        *,
        batch_size: int = 4,
        save_overlay: bool = True,
        alpha: float = 0.5,
        file_extensions: Sequence[str] = DEFAULT_IMAGE_EXTS,
    ) -> None:
        """Run inference over a directory of images and write the outputs to disk."""
        image_paths = self._collect_image_paths(input_dir, file_extensions)
        if not image_paths:
            raise FileNotFoundError(f"No images with extensions {file_extensions} in {input_dir}")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        visualizer = FaceParsingVisualizer() if save_overlay else None

        print(f"Found {len(image_paths)} images to process")
        print(f"Processing with batch size {batch_size}")

        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            predictions = self.predict_batch(
                batch_paths,
                resize_to_original=True,
                output_device="cpu",
            )
            self._write_predictions(batch_paths, predictions, output_path, visualizer, alpha)

    def _load_model(
        self,
        checkpoint_path: Union[str, Path],
        dinov3_checkpoint_path: Optional[Union[str, Path]],
    ) -> FaceParsingModel:
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        dinov3 = None
        if dinov3_checkpoint_path:
            hub_dir = Path(__file__).resolve().parents[1] / "models"
            dinov3 = torch.hub.load(
                str(hub_dir),
                "dinov3_vitb16",
                source="local",
                weights=str(dinov3_checkpoint_path),
            )
            dinov3 = dinov3.to(self.device)
            dinov3.eval()
            for param in dinov3.parameters():
                param.requires_grad = False
        model = FaceParsingModel(
            num_classes=self.num_classes,
            encoder_device=self.device,
            encoder_dinov3=dinov3,
        )

        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        filtered_state_dict = {k: v for k, v in state_dict.items() if "encoder.dinov3" not in k}
        model.load_state_dict(filtered_state_dict, strict=False)
        model.to(self.device)
        model.eval()
        return model

    @staticmethod
    def _build_default_transform(image_size: Tuple[int, int]) -> transforms.Compose:
        """Return the default preprocessing pipeline."""
        return transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _prepare_image(
        self,
        sample: ImageLike,
    ) -> dict:
        """Convert any supported input into a normalized tensor."""
        pil_image, source_path = self._to_pil(sample)
        tensor = self.transform(pil_image)
        return {
            "tensor": tensor,
            "original_size": pil_image.size,
            "source_path": source_path,
        }

    def _to_pil(self, sample: ImageLike) -> Tuple[Image.Image, Optional[str]]:
        if isinstance(sample, (str, Path)):
            path = str(sample)
            image = Image.open(path).convert("RGB")
            return image, path
        if isinstance(sample, Image.Image):
            return sample.convert("RGB"), None
        if isinstance(sample, np.ndarray):
            array = self._coerce_array(sample)
            return Image.fromarray(array), None
        if torch.is_tensor(sample):
            tensor = sample.detach().cpu()
            if tensor.dim() == 4 and tensor.size(0) == 1:
                tensor = tensor.squeeze(0)
            if tensor.dim() != 3:
                raise ValueError("Tensor inputs must have shape (C, H, W) or (1, C, H, W).")
            return TF.to_pil_image(tensor), None
        raise TypeError(f"Unsupported input type: {type(sample)}")

    @staticmethod
    def _coerce_array(array: np.ndarray) -> np.ndarray:
        array = np.asarray(array)
        if array.ndim == 2:
            array = np.stack([array] * 3, axis=-1)
        if array.shape[-1] == 4:
            array = array[..., :3]
        if array.dtype != np.uint8:
            if array.max() <= 1.0:
                array = (array * 255.0).clip(0, 255)
            else:
                array = array.clip(0, 255)
            array = array.astype(np.uint8)
        return array

    def _resize_masks(
        self,
        masks: torch.Tensor,
        original_sizes: Sequence[Tuple[int, int]],
    ) -> List[torch.Tensor]:
        """Resize predicted masks back to the original image sizes."""
        resized: List[torch.Tensor] = []
        for idx, (width, height) in enumerate(original_sizes):
            mask_tensor = masks[idx].unsqueeze(0).unsqueeze(0).float()
            resized_mask = F.interpolate(
                mask_tensor,
                size=(height, width),
                mode="nearest",
            ).squeeze(0).squeeze(0).long()
            resized.append(resized_mask)
        return resized

    def _collect_image_paths(
        self,
        input_dir: Union[str, Path],
        file_extensions: Sequence[str],
    ) -> List[Path]:
        path = Path(input_dir)
        if path.is_file():
            return [path]

        collected: List[Path] = []
        lower_exts = {ext.lower() for ext in file_extensions}
        for file in sorted(path.iterdir()):
            if file.suffix.lower() in lower_exts:
                collected.append(file)
        return collected

    def _write_predictions(
        self,
        image_paths: Sequence[Path],
        predictions: Sequence[FaceParsingSingleResult],
        output_dir: Path,
        visualizer: Optional[FaceParsingVisualizer],
        alpha: float,
    ) -> None:
        for path, prediction in zip(image_paths, predictions):
            base_name = Path(path).stem
            mask_np = prediction.numpy_mask()
            mask_image = Image.fromarray(mask_np.astype(np.uint8), mode="L")
            mask_path = output_dir / f"{base_name}_mask.png"
            mask_image.save(mask_path)

            if visualizer and prediction.source_path:
                visualizer.visualize_prediction(
                    prediction.source_path,
                    prediction.mask_resized if prediction.mask_resized is not None else prediction.mask,
                    alpha=alpha,
                    save_path=str(output_dir / f"{base_name}_overlay.jpg"),
                )


def load_face_parsing_predictor(**kwargs) -> FaceParsingPredictor:
    """Factory helper mirroring the class signature for convenience."""
    return FaceParsingPredictor(**kwargs)
