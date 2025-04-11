# DepthFlow on Replicate

This is a Replicate implementation of [DepthFlow](https://github.com/BrokenSource/DepthFlow), an advanced image-to-video converter that transforms static pictures into 3D parallax animations.

## Model Description

DepthFlow creates stunning 3D parallax animations from static images by:
1. Generating or using a provided depth map using state-of-the-art depth estimation models
2. Applying advanced GLSL shaders for the parallax effect through ModernGL
3. Producing high-quality video output with configurable parameters

## Input Parameters

- `image`: The source image to create the parallax effect from
- `depth_map` (optional): A custom depth map image. If not provided, one will be generated using DepthAnything V2
- `animation_type`: Type of animation to apply (choices: vertical, horizontal, zoom, circle, dolly, orbital)
- `duration`: Length of the output video in seconds (default: 3.0, range: 1.0-10.0)
- `fps`: Frames per second for the output video (default: 30, range: 15-60)
- `parallax_intensity`: Strength of the parallax effect (default: 1.0, range: 0.1-5.0)

## Output

The model outputs an MP4 video file containing the parallax animation.

## Example Usage

```python
import replicate

output = replicate.run(
    "username/depthflow:version",
    input={
        "image": "path/to/image.jpg",
        "animation_type": "orbital",
        "duration": 3.0,
        "fps": 30,
        "parallax_intensity": 1.0
    }
)
```

## Animation Types

1. **Vertical**: Up and down camera motion
2. **Horizontal**: Left and right camera motion
3. **Zoom**: Forward and backward camera motion
4. **Circle**: Circular camera motion around the subject
5. **Dolly**: Dolly zoom effect (Vertigo effect)
6. **Orbital**: Orbiting camera motion around a fixed point

## References

- Original DepthFlow repository: https://github.com/BrokenSource/DepthFlow
- Replicate documentation: https://replicate.com/docs
