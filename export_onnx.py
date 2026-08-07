"""Export a trained checkpoint to ONNX.

Usage (on the cluster, where the checkpoints live):

    conda activate ty
    python export_onnx.py -f <ckpt> -o onnx/LLAT_polar_vtvr_v1.onnx \
                          --yaml onnx/LLAT_polar_vtvr_v1.yaml

The input shape is derived from the checkpoint's own hyperparameters rather than
written out by hand. The previous version carried five commented-out
`input_sample` lines and used a Cartesian 81x81 shape, which for a polar model
either fails inside the exporter or - worse - succeeds against a graph that
expects something else. `save_hyperparameters()` already records
data_spatial_shape and the variable lists, so the checkpoint is the single source
of truth.

Two optional safeguards, both on by default:

  --yaml   cross-check the shape and variable lists against the model card the
           inference wrapper will read. Export and inference drifting apart is
           exactly the failure S2/S3 were about, and it is silent.
  --verify run the exported graph through onnxruntime and compare against
           PyTorch. An ONNX export can quietly produce a different graph -
           traced control flow, a dropped op - and nothing downstream would
           report it; the forecast would just be wrong.
"""
import argparse
import os

import torch

from models.lightning_modules import PanguLightningModule

TOL = 1e-4


def derive_input_sample(model):
    """Build (upper, surface) dummy inputs from the checkpoint's hyperparameters."""
    hp = model.hparams
    z, r, theta = hp.data_spatial_shape
    n_upper = len(hp.upper_vars)
    n_surface = len(hp.surface_vars)
    return (torch.randn(1, z, r, theta, n_upper),
            torch.randn(1, r, theta, n_surface))


def check_against_yaml(model, yaml_path):
    """Fail loudly if the model card disagrees with the checkpoint."""
    import yaml as _yaml

    with open(yaml_path, encoding='utf-8') as f:
        card = _yaml.safe_load(f)
    hp = model.hparams

    polar = card.get('polar')
    if polar is None:
        raise KeyError(f"{yaml_path} has no `polar:` block")

    problems = []
    if list(polar['data_spatial_shape']) != list(hp.data_spatial_shape):
        problems.append(f"data_spatial_shape: yaml {polar['data_spatial_shape']} "
                        f"vs checkpoint {list(hp.data_spatial_shape)}")
    if list(card['upper_vars']) != list(hp.upper_vars):
        problems.append(f"upper_vars: yaml {card['upper_vars']} "
                        f"vs checkpoint {list(hp.upper_vars)}")
    if list(card['surface_vars']) != list(hp.surface_vars):
        problems.append(f"surface_vars: yaml {card['surface_vars']} "
                        f"vs checkpoint {list(hp.surface_vars)}")
    if problems:
        raise ValueError(
            f"{yaml_path} does not describe this checkpoint:\n  "
            + "\n  ".join(problems))
    print(f"[ok] {yaml_path} matches the checkpoint")


def verify(output, model, sample):
    """Compare the exported graph against PyTorch on the same input."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("[skip] onnxruntime is not installed; exported graph NOT verified")
        return

    with torch.no_grad():
        ref_u, ref_s = model(*sample)

    sess = ort.InferenceSession(output, providers=["CPUExecutionProvider"])
    got_u, got_s = sess.run(None, {
        "input_upper": sample[0].numpy(),
        "input_surface": sample[1].numpy(),
    })

    du = float((torch.from_numpy(got_u) - ref_u).abs().max())
    ds = float((torch.from_numpy(got_s) - ref_s).abs().max())
    print(f"[verify] max|onnx - torch|: upper {du:.3e}, surface {ds:.3e}")
    if du > TOL or ds > TOL:
        raise ValueError(
            f"exported graph disagrees with PyTorch beyond {TOL:g}. The ONNX file "
            "should not be used: inference would silently differ from training.")
    print("[ok] exported graph matches PyTorch")


def main(ckpt, output, yaml=None, verify_export=True):
    model = PanguLightningModule.load_from_checkpoint(ckpt, map_location="cpu")
    # eval() matters: DropPath is stochastic in train mode, so a traced export
    # would bake in one particular random path.
    model.eval()

    hp = model.hparams
    sample = derive_input_sample(model)
    print(f"checkpoint : {ckpt}")
    print(f"grid       : (Z,R,Theta) = {tuple(hp.data_spatial_shape)}")
    print(f"variables  : {len(hp.upper_vars)} upper, {len(hp.surface_vars)} surface")
    print(f"input_upper   {tuple(sample[0].shape)}")
    print(f"input_surface {tuple(sample[1].shape)}")

    if yaml:
        check_against_yaml(model, yaml)

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    model.to_onnx(
        output,
        input_sample=sample,
        export_params=True,
        input_names=["input_upper", "input_surface"],
        output_names=["output_upper", "output_surface"],
    )
    print(f"[ok] wrote {output} ({os.path.getsize(output)/1e6:.0f} MB)")

    if verify_export:
        verify(output, model, sample)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-f', "--ckpt", type=str, required=True,
                        help="checkpoint to export")
    parser.add_argument('-o', "--output", type=str, required=True,
                        help="destination .onnx path")
    parser.add_argument("--yaml", type=str, default=None,
                        help="model card to cross-check against (recommended)")
    parser.add_argument("--no-verify", dest="verify_export", action="store_false",
                        help="skip the PyTorch/ONNX comparison")
    args = parser.parse_args()

    main(**vars(args))
