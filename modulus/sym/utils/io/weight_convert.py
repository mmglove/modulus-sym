import paddle
try:
    import torch
except ModuleNotFoundError:
    pass

import numpy as np

from typing import Dict
import os
import os.path as osp


def torch_to_paddle(torch_state_dict: Dict[str, "torch.Tensor"], output_dir: str, filename: str) -> None:
    """Convert torch state_dict to paddle state_dict and save it to given path.

    Args:
        torch_state_dict (Dict[str, torch.Tensor]): Torch state_dict.
        output_dir (str): Output directory.
        filename (str): Output file name.
    """

    if not filename.endswith(".pdparams"):
        raise ValueError(
            f"filename should ends with '.pdparams', but got {filename}"
        )

    dump_path = osp.join(output_dir, filename)
    if osp.exists(dump_path):
        print(f"✨ ✨ Skip converting as {dump_path} already exist.")
        return

    torch_dump_path = dump_path.replace(".pdparams", ".pth")
    os.makedirs(output_dir, exist_ok=True)
    torch.save(torch_state_dict, torch_dump_path)
    print(f"✨ ✨ torch weights has been saved to: {torch_dump_path} for backup.")

    paddle_state_dict: Dict[str, "paddle.Tensor"] = {}

    for k, v in torch_state_dict.items():
        v_numpy: np.ndarray = v.detach().cpu().numpy()

        if k.endswith("linear.weight"):
            assert v_numpy.ndim == 2, (
                f"ndim of v_numpy should be 2, but got {v_numpy.ndim}."
            )
            if 'final_layer' in k or 'output_linear' in k:
                paddle_state_dict[k] = v_numpy.T
                print("✨ ✨ tranpose weight created by nn.Linear.")
            else:
                paddle_state_dict[k] = v_numpy
        elif ('fc' in k and 'weight' in k):
            paddle_state_dict[k] = v_numpy.T
        else:
            paddle_state_dict[k] = v_numpy

    paddle_state_dict = {
        k: paddle.to_tensor(v)
        for k, v in paddle_state_dict.items()
    }

    os.makedirs(output_dir, exist_ok=True)
    paddle.save(paddle_state_dict, dump_path)
    print(f"✨ ✨ .pth weights has been converted tp .pdparams and saved to: {dump_path}")
