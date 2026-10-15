"""Small kernel execution check, not a training pilot or benchmark."""
import json
import torch
from torch import nn
from .models import configure_fp8


def main():
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU attached to this process.")
    device = torch.device("cuda")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("GPU does not support the required BF16 mixed-precision operations.")
    layer = nn.Sequential(nn.Linear(128, 256), nn.GELU(), nn.Linear(256, 128)).to(device)
    info = configure_fp8(layer, "fp8", device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        result = layer(torch.randn(32, 128, device=device))
        loss = result.float().square().mean()
    loss.backward()
    torch.cuda.synchronize()
    if not torch.isfinite(loss) or any(not torch.isfinite(p.grad).all() for p in layer.parameters()):
        raise SystemExit("FP8 kernel check produced non-finite output/gradients.")
    print(json.dumps({"gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
                      "cuda": torch.version.cuda, "fp8_forward_backward": "passed",
                      **info}, indent=2))


if __name__ == "__main__":
    main()
