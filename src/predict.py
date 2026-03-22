import argparse
from pathlib import Path

import torch
from PIL import Image

from src.data_loader import get_transforms
from src.model import build_model


CLASS_NAMES = ["cat", "dog"]


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on a single animal image.")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="artifacts/models/best_model.pth",
        help="Path to the trained model checkpoint",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Input image size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use",
    )
    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path: str, device: torch.device, num_classes: int = 2):
    model = build_model(num_classes=num_classes, pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location=device)

    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path: str, image_size: int = 224) -> torch.Tensor:
    _, eval_transform = get_transforms(image_size=image_size)

    image = Image.open(image_path).convert("RGB")
    image_tensor = eval_transform(image).unsqueeze(0)
    return image_tensor


def predict_image(
    image_path: str,
    checkpoint_path: str = "artifacts/models/best_model.pth",
    image_size: int = 224,
    device: str = "auto",
):
    resolved_device = get_device(device)
    model = load_model(
        checkpoint_path=checkpoint_path,
        device=resolved_device,
        num_classes=len(CLASS_NAMES),
    )
    image_tensor = preprocess_image(image_path=image_path, image_size=image_size).to(
        resolved_device
    )

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)

    predicted_index = int(torch.argmax(probabilities).item())
    confidence = float(probabilities[predicted_index].item())
    probabilities_by_class = {
        class_name: float(probabilities[index].item())
        for index, class_name in enumerate(CLASS_NAMES)
    }

    return {
        "label": CLASS_NAMES[predicted_index],
        "confidence": confidence,
        "probabilities": probabilities_by_class,
    }


def main():
    args = parse_args()

    image_path = Path(args.image_path)
    checkpoint_path = Path(args.checkpoint)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    result = predict_image(
        image_path=str(image_path),
        checkpoint_path=str(checkpoint_path),
        image_size=args.image_size,
        device=args.device,
    )

    print(f"Image: {image_path}")
    print(f"Predicted label: {result['label']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print("Class probabilities:")
    for class_name, probability in result["probabilities"].items():
        print(f"  {class_name}: {probability:.4f}")


if __name__ == "__main__":
    main()
