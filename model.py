# model.py
#
# Manual GRU forward pass for MicroPython.
#
# The model receives a historical acceleration window:
#     [ax, ay, az]
#
# It produces:
#     [ax_next, ay_next, az_next,
#      vx_next, vy_next, vz_next]
import math
from array import array

import weights_meta as meta
import gc
import struct
INPUT_SIZE = meta.INPUT_SIZE
HIDDEN_SIZE = meta.HIDDEN_SIZE
NUM_LAYERS = meta.NUM_LAYERS
OUTPUT_SIZE = meta.OUTPUT_SIZE
WINDOW_SIZE = meta.WINDOW_SIZE


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def _load_weights(path="weights.bin"):
    """
    Load float32 GRU weights from weights.bin.

    This implementation is compatible with MicroPython:
    - It does not use array.frombytes().
    - It reads one float32 value at a time.
    - It avoids loading the complete file into RAM.
    """

    hidden_size = HIDDEN_SIZE

    layout = []

    for layer_index in range(NUM_LAYERS):
        if layer_index == 0:
            layer_input_size = INPUT_SIZE
        else:
            layer_input_size = HIDDEN_SIZE

        # PyTorch GRU parameter order:
        # weight_ih, weight_hh, bias_ih, bias_hh
        layout.append(3 * hidden_size * layer_input_size)
        layout.append(3 * hidden_size * hidden_size)
        layout.append(3 * hidden_size)
        layout.append(3 * hidden_size)

    # Final fully connected layer
    layout.append(OUTPUT_SIZE * hidden_size)
    layout.append(OUTPUT_SIZE)

    tensors = []

    with open(path, "rb") as file:
        for tensor_index, float_count in enumerate(layout):
            tensor = array("f")

            for _ in range(float_count):
                raw_value = file.read(4)

                if len(raw_value) != 4:
                    raise ValueError(
                        "weights.bin ended unexpectedly while "
                        "loading tensor {}".format(tensor_index)
                    )

                value = struct.unpack("<f", raw_value)[0]
                tensor.append(value)

            tensors.append(tensor)

            # Release temporary objects between tensors
            gc.collect()

    return tensors




_W = _load_weights()

_LAYER_W = []
_idx = 0
for _l in range(NUM_LAYERS):
    _LAYER_W.append((_W[_idx], _W[_idx + 1], _W[_idx + 2], _W[_idx + 3]))
    _idx += 4
_FC_WEIGHT = _W[_idx]
_FC_BIAS = _W[_idx + 1]


def _matvec(mat, ncols, vec, out, row_count):
    """out[r] = dot(mat[r, :], vec) for r in range(row_count). mat is flat, row-major."""
    for r in range(row_count):
        s = 0.0
        base = r * ncols
        for c in range(ncols):
            s += mat[base + c] * vec[c]
        out[r] = s


def _gru_cell(x, h_prev, w_ih, w_hh, b_ih, b_hh, in_size, hidden_size):
    gi = [0.0] * (3 * hidden_size)
    gh = [0.0] * (3 * hidden_size)
    _matvec(w_ih, in_size, x, gi, 3 * hidden_size)
    _matvec(w_hh, hidden_size, h_prev, gh, 3 * hidden_size)

    h_new = [0.0] * hidden_size
    for i in range(hidden_size):
        i_r = gi[i] + b_ih[i]
        i_z = gi[hidden_size + i] + b_ih[hidden_size + i]
        i_n = gi[2 * hidden_size + i] + b_ih[2 * hidden_size + i]

        h_r = gh[i] + b_hh[i]
        h_z = gh[hidden_size + i] + b_hh[hidden_size + i]
        h_n = gh[2 * hidden_size + i] + b_hh[2 * hidden_size + i]

        r = _sigmoid(i_r + h_r)
        z = _sigmoid(i_z + h_z)
        n = math.tanh(i_n + r * h_n)
        h_new[i] = (1.0 - z) * n + z * h_prev[i]
    return h_new


def predict(window):
    """
    Predict next-step acceleration and velocity.
    Input:
        window:
            List containing WINDOW_SIZE acceleration samples.

            Each sample:
                (ax, ay, az)

            All inputs must use the same normalization
            applied during model training.
    Return order:
        [
            ax_next,
            ay_next,
            az_next,
            vx_next,
            vy_next,
            vz_next
        ]
    """
    h_states = [
        [0.0] * HIDDEN_SIZE
        for _ in range(NUM_LAYERS)
    ]

    for time_index in range(WINDOW_SIZE):

        layer_input = window[time_index]

        for layer_index in range(NUM_LAYERS):

            w_ih, w_hh, b_ih, b_hh = (
                _LAYER_W[layer_index]
            )

            if layer_index == 0:
                input_size = INPUT_SIZE
            else:
                input_size = HIDDEN_SIZE

            h_states[layer_index] = _gru_cell(
                layer_input,
                h_states[layer_index],
                w_ih,
                w_hh,
                b_ih,
                b_hh,
                input_size,
                HIDDEN_SIZE
            )

            layer_input = h_states[layer_index]

    final_hidden = h_states[NUM_LAYERS - 1]

    relu_hidden = [
        value if value > 0.0 else 0.0
        for value in final_hidden
    ]

    outputs = [0.0] * OUTPUT_SIZE

    for output_index in range(OUTPUT_SIZE):

        result = _FC_BIAS[output_index]

        weight_start = (
            output_index * HIDDEN_SIZE
        )

        for hidden_index in range(HIDDEN_SIZE):

            result += (
                _FC_WEIGHT[
                    weight_start + hidden_index
                ]
                * relu_hidden[hidden_index]
            )

        outputs[output_index] = result

    return outputs