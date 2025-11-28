# Intermediate Frame Interpolation Underwater 3D Gaussian Splatting Documentation

## 1. Environment Configuration

- **Python Version**: *3.12.9*
- **CUDA Version**: *12.8* 
- **PyTorch Version**: *2.5.1*
- **Torchvision Version**: *0.20.1*
- **Torchaudio Version**: *2.5.1*

---

## 2. Previous Documentation

The original `README.md` has been renamed to [`README_OLD.md`](README_OLD.md).

---

## 3. New Contributions

- **Deeper MLP Network**: [nerf_model.py](scene/nerf_model.py)
- **Adaptive Bilateral Filter**: [render.py](render.py)
- **Decoupling Learning of the RGB Channel**: [nerf_model.py](scene/nerf_model.py)
- **Intermediate Frame Interpolation**: Please refer to the [Practical-RIFE GitHub Repository](https://github.com/hzwer/Practical-RIFE)

---

## 4. New Flags

- `--frame`
  - **Purpose**: Set the frame weight for interpolated frames.
  - **Note**: If `--adaptive` is used, this flag will be ignored.

- `--adaptive`
  - **Purpose**: Use CLIP-IQA to adaptively set weights to the interpolated frame.
  - **Note**: Experimental feature, and it may not guarantee to obtain the best performance.

---

## 5. Frame Interpolation

**Important**: When generating the interpolated frame, the filenames of the frame must include `_to_`. This is the flag to distinguish the interpolated frame from the original frame.

---

## 6. Running the Code

Please refer to the instructions provided in [`README_OLD.md`](README_OLD.md) for detailed setup and usage guidelines.

---
