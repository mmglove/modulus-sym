# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import paddle
from typing import List, Optional
from modulus.sym.key import Key
from modulus.sym.constants import diff
from modulus.sym.eq.derivatives import Derivative


class Model(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, z):
        return (
            1.5 * x * x + paddle.sin(y) + paddle.exp(z),
            2 * x * x + paddle.cos(y) + paddle.exp(-z),
            1.5 * x * x + paddle.sin(y) + paddle.exp(z),
            2 * x * x + paddle.cos(y) + paddle.exp(-z),
        )


def validate_gradients(
    x, y, z, dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz, dpdx, dpdy, dpdz
):
    # Check against exact solution
    assert paddle.allclose(dudx, 3 * x).item(), "x derivative of u failed"
    assert paddle.allclose(
        dudy, paddle.cos(y)
    ).item(), "y derivative of u  failed"
    assert paddle.allclose(
        dudz, paddle.exp(z)
    ).item(), "z derivative of u  failed"
    assert paddle.allclose(dvdx, 4 * x).item(), "x derivative of v failed"
    assert paddle.allclose(
        dvdy, -paddle.sin(y)
    ).item(), "y derivative of v failed"
    assert paddle.allclose(
        dvdz, -paddle.exp(-z)
    ).item(), "z derivative of v failed"
    assert paddle.allclose(dwdx, 3 * x).item(), "x derivative of w failed"
    assert paddle.allclose(dwdy, paddle.cos(y)).item(), "y derivative of w failed"
    assert paddle.allclose(dwdz, paddle.exp(z)).item(), "z derivative of w failed"
    assert paddle.allclose(dpdx, 4 * x).item(), "x derivative of p failed"
    assert paddle.allclose(
        dpdy, -paddle.sin(y)
    ).item(), "y derivative of p failed"
    assert paddle.allclose(
        dpdz, -paddle.exp(-z)
    ).item(), "z derivative of p failed"


def test_derivative_node():
    device = str("cuda:0" if paddle.device.cuda.device_count() >= 1 else "cpu").replace(
        "cuda", "gpu"
    )
    # Set up input coordinates
    batch_size = 128
    x = paddle.rand(shape=[batch_size, 1], dtype="float32").to(device)
    x.stop_gradient = not True
    y = paddle.rand(shape=[batch_size, 1], dtype="float32").to(device)
    y.stop_gradient = not True
    z = paddle.rand(shape=[batch_size, 1], dtype="float32").to(device)
    z.stop_gradient = not True

    # Instantiate the model and compute outputs
    model = Model()
    u, v, w, p = model(x, y, z)

    input_vars = [
        Key.from_str("x"),
        Key.from_str("y"),
        Key.from_str("z"),
        Key.from_str("u"),
        Key.from_str("v"),
        Key.from_str("w"),
        Key.from_str("p"),
    ]
    derivs = [
        Key.from_str(diff("u", "x")),
        Key.from_str(diff("u", "y")),
        Key.from_str(diff("u", "z")),
        Key.from_str(diff("v", "x")),
        Key.from_str(diff("v", "y")),
        Key.from_str(diff("v", "z")),
        Key.from_str(diff("w", "x")),
        Key.from_str(diff("w", "y")),
        Key.from_str(diff("w", "z")),
        Key.from_str(diff("p", "x")),
        Key.from_str(diff("p", "y")),
        Key.from_str(diff("p", "z")),
    ]
    dnode = Derivative.make_node(input_vars, derivs, jit=False)

    input_dict = dict(zip((str(v) for v in input_vars), [x, y, z, u, v, w, p]))
    derivs_dict = dnode.evaluate(input_dict)
    validate_gradients(x, y, z, *(derivs_dict[str(d)] for d in derivs))


if __name__ == "__main__":
    test_derivative_node()
