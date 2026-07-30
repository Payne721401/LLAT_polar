import argparse

import torch

from models.lightning_modules import PanguLightningModule


def create_model_pl_module(ckpt_path: str) -> PanguLightningModule:
    model_pl_module = PanguLightningModule.load_from_checkpoint(
        ckpt_path,
    )
    return model_pl_module


def main(
    ckpt: str,
    output: str,
) -> None:
    model = PanguLightningModule.load_from_checkpoint(ckpt)

    model.to_onnx(
        output,
        # input_sample=(torch.randn(1, 13, 81, 81, 6), torch.randn(1, 81, 81, 10)),
        input_sample=(torch.randn(1, 13, 81, 81, 6), torch.randn(1, 81, 81, 20)),
        # input_sample=(torch.randn(1, 13, 81, 81, 10), torch.randn(1, 81, 81, 22)),
        # input_sample=(torch.randn(1, 13, 161, 161, 5), torch.randn(1, 161, 161, 4)),
        # input_sample=(torch.randn(1, 13, 161, 161, 6), torch.randn(1, 161, 161, 7)),
        export_params=True,
        input_names=["input_upper", "input_surface"],
        output_names=["output_upper", "output_surface"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f',"--ckpt", type=str, required=True)
    parser.add_argument('-o',"--output", type=str, required=True)
    args = parser.parse_args()

    main(**vars(args))
