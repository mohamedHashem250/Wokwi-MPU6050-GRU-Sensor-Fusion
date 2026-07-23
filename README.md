# GRU-Based IMU Sensor Fusion and Velocity Prediction

**Final Project — AI-Driven Cyber-Physical Systems, Summer 2026**

**Prepared by:** Mohamed Abouhashem Salama  
PhD Student, University of South Alabama  

**Instructor and Supervisor:** Dr. Kolitha Warnakulasooriya

---

## Project Mission

This project receives three-axis acceleration measurements from an MPU6050 IMU and uses a compact GRU network to process a rolling window of 30 historical samples.

The model produces six values:

- Fused next-step acceleration: `ax`, `ay`, `az`
- Predicted next-step velocity: `vx`, `vy`, `vz`

The trained model is deployed on a simulated Raspberry Pi Pico using MicroPython and the Wokwi simulator.

---

## Assignment Requirement

The system must:

1. Receive acceleration data from the IMU:
   - X-axis
   - Y-axis
   - Z-axis
2. Fuse the historical measurements.
3. Print the fused information.
4. Produce velocity values for the X, Y, and Z axes.

This implementation extends the required output by predicting both fused acceleration and velocity on all three axes.

---

## Project Workflow

```text
MPU6050 acceleration
        ↓
30-sample historical window
        ↓
Compact GRU model
        ↓
Fused acceleration: ax, ay, az
Predicted velocity: vx, vy, vz
```

The Raspberry Pi Pico reads the MPU6050 through I2C, normalizes the measurements, stores the latest 30 samples, and performs a manual GRU forward pass without NumPy or PyTorch on the device.

---

## Hardware

- Raspberry Pi Pico with RP2040, simulated in Wokwi
- MPU6050 IMU accelerometer
- I2C connections:
  - `GP0` → SDA
  - `GP1` → SCL
- Raspberry Pi Pico SRAM: 264 KB

---

## Software

- Visual Studio Code
- Wokwi VS Code extension
- MicroPython v1.27.0
- Python
- PyTorch
- NumPy
- pandas
- Matplotlib
- Jupyter Notebook
- `mpremote`

---

## Compact GRU Architecture

| Component | Configuration |
|---|---|
| Input features | 3: `ax`, `ay`, `az` |
| Historical window | 30 samples |
| GRU layers | 1 |
| Hidden size | 32 |
| Activation | ReLU |
| Output layer | Linear `32 → 6` |
| Output values | `ax`, `ay`, `az`, `vx`, `vy`, `vz` |
| Total parameters | 3,750 |
| Float32 weight size | Approximately 15,000 bytes |

The compact network was selected because the Raspberry Pi Pico has limited SRAM that must also hold the MicroPython runtime, Python objects, hidden states, input buffers, and temporary GRU calculations.

---

## Repository Structure

| File | Description |
|---|---|
| `train_gru_model.ipynb` | Dataset preparation, exploratory analysis, model training, testing, and checkpoint creation |
| `gru_imu_model.pth` | Trained PyTorch checkpoint |
| `export_to_micropython.py` | Converts the PyTorch checkpoint into MicroPython deployment files |
| `weights.bin` | Compact float32 GRU weights |
| `weights_meta.py` | Model dimensions and window configuration |
| `main.py` | Reads the IMU, maintains the input window, runs inference, and prints the results |
| `model.py` | Manual MicroPython-compatible GRU forward pass |
| `imu.py` | MPU6050 communication driver |
| `vector3d.py` | Three-axis vector handling |
| `diagram.json` | Wokwi circuit configuration |
| `wokwi.toml` | Wokwi firmware and RFC2217 configuration |
| `requirements.txt` | Desktop Python dependencies |

---

## 1. Create the Python Environment

Create and activate a virtual environment.

### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

### Linux or macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

---

## 2. Train the GRU Model

Open and run:

```text
train_gru_model.ipynb
```

The notebook performs:

1. Dataset loading and exploratory analysis
2. Data preparation
3. Construction of 30-sample acceleration windows
4. Compact GRU training
5. Model evaluation
6. Checkpoint saving

The model input is:

```text
[ax, ay, az] × 30 historical samples
```

The target order is:

```text
[ax_next, ay_next, az_next, vx_next, vy_next, vz_next]
```

---

## 3. Export the Model for MicroPython

Run the exporter on the computer:

```bash
python export_to_micropython.py
```

Expected output:

```text
Wrote weights.bin (15000 bytes, 3750 floats)
Wrote weights_meta.py

input_size=3
hidden_size=32
num_layers=1
output_size=6
window_size=30
```

The exporter creates:

- `weights.bin`
- `weights_meta.py`

The PyTorch checkpoint and training notebook are not uploaded to the Pico.

---

## 4. Configure Wokwi

Use the following `wokwi.toml` configuration:

```toml
[wokwi]
version = 1
firmware = "RPI_PICO-20251209-v1.27.0.uf2"
rfc2217ServerPort = 4000
```

Open `diagram.json` in Visual Studio Code and start the Wokwi simulation.

Wait until the MicroPython terminal shows:

```text
>>>
```

---

## 5. Upload the Deployment Files

Keep the simulator running and open another terminal in the project directory.

Upload the files:

```bash
mpremote connect port:rfc2217://localhost:4000 fs cp main.py :main.py
mpremote connect port:rfc2217://localhost:4000 fs cp imu.py :imu.py
mpremote connect port:rfc2217://localhost:4000 fs cp vector3d.py :vector3d.py
mpremote connect port:rfc2217://localhost:4000 fs cp model.py :model.py
mpremote connect port:rfc2217://localhost:4000 fs cp weights_meta.py :weights_meta.py
mpremote connect port:rfc2217://localhost:4000 fs cp weights.bin :weights.bin
```

---

## 6. Confirm the Uploaded Files

Run:

```bash
mpremote connect port:rfc2217://localhost:4000 ls
```

Expected files:

```text
main.py
imu.py
vector3d.py
model.py
weights_meta.py
weights.bin
```

The files can also be checked from the MicroPython REPL:

```python
import os
print(os.listdir("/"))
```

---

## 7. Run the Project

Press `Ctrl+D` inside the Wokwi MicroPython terminal to perform a soft reset.

MicroPython automatically executes `main.py`. Do not run:

```bash
python main.py
```

The `machine`, `Pin`, and `I2C` modules are provided by MicroPython and are not available in standard desktop Python.

---

## Example Output

During buffering:

```text
Starting Pi Pico application...
Loading compact GRU model...
Compact GRU model loaded successfully.
I2C devices: [104]
MPU6050 initialized successfully.

Measured Accel [g]: x=0.000 y=0.000 z=1.000
Buffering... 1/30
```

After collecting 30 samples:

```text
Measured Accel [g]: x=0.650 y=0.350 z=0.000
Predicted Accel [norm]: x=0.587 y=0.639 z=0.531
Predicted Accel [g]: x=0.350 y=0.556 z=0.125
Predicted Velocity [norm]: x=0.101 y=0.823 z=0.290
Inference Time: 3087.4 ms
```

---

## Output Interpretation

- **Measured Accel [g]**: Current MPU6050 acceleration measurement
- **Predicted Accel [norm]**: Fused next-step acceleration in the normalized training scale
- **Predicted Accel [g]**: Fused next-step acceleration converted back to gravitational units
- **Predicted Velocity [norm]**: Predicted velocity values in the normalized target scale
- **Inference Time**: Time required for one GRU prediction on the simulated Pico

---

## Important Note About Normalization

The live MPU6050 measurements must be normalized using the same method used during model training. A mismatch between training and deployment normalization can reduce the accuracy of both acceleration and velocity predictions.

---

## Summary

This project demonstrates a complete embedded machine-learning workflow:

1. Train a compact GRU model using historical IMU acceleration data.
2. Export the trained PyTorch parameters into a MicroPython-compatible binary format.
3. Deploy the model to a simulated Raspberry Pi Pico.
4. Read real-time MPU6050 acceleration measurements.
5. Predict fused acceleration and velocity for the X, Y, and Z axes.
