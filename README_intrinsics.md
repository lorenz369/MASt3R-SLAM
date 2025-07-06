# Using Intrinsics Files with MASt3R SLAM

MASt3R SLAM now supports loading camera intrinsics from simple text files, allowing you to use accurate camera parameters instead of relying on smartphone estimates.

## Intrinsics File Format

Create a text file with the following format:

```
fx, fy, ox, oy
1476.8237, 1476.8237, 933.1647, 715.0366
```

Where:
- `fx, fy`: Focal lengths in pixels
- `ox, oy`: Principal point coordinates (optical center) in pixels

## Usage Options

### Option 1: Command Line Argument

```bash
python main.py --dataset your_dataset --intrinsics config/IPhone11.txt
```

### Option 2: Configuration File

Create a config file (e.g., `config/iphone11.yaml`):

```yaml
inherit: "config/base.yaml"

intrinsics_file: "config/IPhone11.txt"
use_calib: False  # We use intrinsics file instead of full calibration

dataset:
  subsample: 1
  img_downsample: 1
```

Then run:

```bash
python main.py --dataset your_dataset --config config/iphone11.yaml
```

## How It Works

1. **Loading**: The system loads intrinsics from the specified file
2. **Scaling**: Intrinsics are automatically scaled based on image resizing and downsampling
3. **Integration**: The K matrix is set on each frame, enabling proper camera projection
4. **Fallback**: If no intrinsics file is provided, the system falls back to smartphone estimates

## Example: iPhone 11 Intrinsics

The provided `IPhone11.txt` contains real iPhone 11 camera parameters:

```
fx, fy, ox, oy
1476.8237, 1476.8237, 933.1647, 715.0366
```

These parameters represent:
- Image resolution: ~1866×1430 pixels
- Field of view: Similar to typical smartphone cameras
- fx/width ratio: ~0.79 (within smartphone range of 0.8-0.9)

## Comparison with Estimates

**With intrinsics file**: Uses your actual camera parameters
**Without intrinsics file**: Uses smartphone estimates (fx = 0.85 × max_dimension)

The intrinsics file approach provides more accurate 3D reconstruction and camera tracking, especially for calibrated cameras or when precise measurements are available.

## Technical Details

- Intrinsics are automatically transformed to account for MASt3R's image resizing pipeline
- The system applies the same scaling logic as the built-in calibration system
- Both focal lengths and principal point are properly adjusted for downsampling and cropping
- No distortion parameters are currently supported in the simple text format

## Creating Your Own Intrinsics File

1. **From camera calibration**: Use OpenCV camera calibration or similar tools
2. **From smartphone specs**: Look up your device's camera specifications
3. **From EXIF data**: Extract focal length and sensor size information
4. **From existing datasets**: Many datasets provide camera intrinsics

Save the values in the simple two-line format shown above. 