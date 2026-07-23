import time

time.sleep(0.1)

print("Starting Pi Pico application...")

from machine import I2C, Pin
from imu import MPU6050

print("Loading compact GRU model...")

import model
from weights_meta import WINDOW_SIZE

print("Compact GRU model loaded successfully.")


# -------------------------------------------------
# I2C and MPU6050 initialization
# -------------------------------------------------

i2c = I2C(
    0,
    scl=Pin(1),
    sda=Pin(0),
    freq=400000
)

print("I2C devices:", i2c.scan())

imu = MPU6050(i2c)

print("MPU6050 initialized successfully.")


# -------------------------------------------------
# Acceleration normalization
# -------------------------------------------------

ACCEL_MIN_G = -2.0
ACCEL_MAX_G = 2.0


def normalize_acceleration(value_g):
    """
    Map acceleration from [-2g, +2g] to [0, 1].
    """

    normalized_value = (
        value_g - ACCEL_MIN_G
    ) / (
        ACCEL_MAX_G - ACCEL_MIN_G
    )

    if normalized_value < 0.0:
        return 0.0

    if normalized_value > 1.0:
        return 1.0

    return normalized_value


def denormalize_acceleration(normalized_value):
    """
    Map normalized acceleration from [0, 1]
    back to acceleration in g.
    """

    return (
        normalized_value
        * (ACCEL_MAX_G - ACCEL_MIN_G)
        + ACCEL_MIN_G
    )


# -------------------------------------------------
# Historical window
# -------------------------------------------------

window = []


# -------------------------------------------------
# Main acquisition and inference loop
# -------------------------------------------------

while True:

    ax, ay, az = imu.accel.xyz

    print(
        "Measured Accel [g]: "
        "x={:.3f} y={:.3f} z={:.3f}".format(
            ax,
            ay,
            az
        )
    )

    normalized_ax = normalize_acceleration(ax)
    normalized_ay = normalize_acceleration(ay)
    normalized_az = normalize_acceleration(az)

    window.append(
        (
            normalized_ax,
            normalized_ay,
            normalized_az
        )
    )

    if len(window) > WINDOW_SIZE:
        window.pop(0)

    if len(window) == WINDOW_SIZE:

        start_time = time.ticks_us()

        predictions = model.predict(window)

        inference_time_ms = (
            time.ticks_diff(
                time.ticks_us(),
                start_time
            )
            / 1000
        )

        (
            fused_ax_norm,
            fused_ay_norm,
            fused_az_norm,
            predicted_vx,
            predicted_vy,
            predicted_vz
        ) = predictions

        fused_ax_g = denormalize_acceleration(
            fused_ax_norm
        )

        fused_ay_g = denormalize_acceleration(
            fused_ay_norm
        )

        fused_az_g = denormalize_acceleration(
            fused_az_norm
        )

        print(
            "Predicted Accel [norm]: "
            "x={:.3f} y={:.3f} z={:.3f}".format(
                fused_ax_norm,
                fused_ay_norm,
                fused_az_norm
            )
        )

        print(
            "Predicted Accel [g]: "
            "x={:.3f} y={:.3f} z={:.3f}".format(
                fused_ax_g,
                fused_ay_g,
                fused_az_g
            )
        )

        print(
            "Predicted Velocity [norm]: "
            "x={:.3f} y={:.3f} z={:.3f}".format(
                predicted_vx,
                predicted_vy,
                predicted_vz
            )
        )

        print(
            "Inference Time: {:.1f} ms".format(
                inference_time_ms
            )
        )

    else:

        print(
            "Buffering... {}/{}".format(
                len(window),
                WINDOW_SIZE
            )
        )

    time.sleep(0.2)