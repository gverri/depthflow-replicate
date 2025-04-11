# DepthFlow on Replicate

This is an implementation of the [DepthFlow](https://github.com/BrokenSource/DepthFlow) project on Replicate, providing 3D parallax animations from static images.

## Features

This implementation includes:

1. EGL context creation for headless GPU rendering
2. Multiple depth estimation models (DepthAnything, Marigold, ZoeDepth, etc.)
3. Various animation presets (orbital, dolly, zoom, focus, parallax)
4. Output in GIF or MP4 format
5. Configurable animation parameters

## Input Parameters

- `image`: Input image for applying the 3D effect
- `depth_model`: Depth estimation model (choices: anything2, anything1, depthpro, marigold, zoedepth)
- `animation_preset`: Animation style (choices: orbital, dolly, zoom, focus, parallax)
- `depth_strength`: Strength of the depth effect (default: 0.5, range: 0.1-1.0)
- `frames`: Number of frames to render (default: 48, range: 24-120)
- `fps`: Frames per second (default: 24, range: 12-60)
- `output_format`: Output format (choices: mp4, gif)

## How It Works

1. The input image is processed by a depth estimation model
2. The DepthFlow library creates a 3D scene from the image and depth map
3. The selected animation preset is applied to the scene
4. Frames are rendered using GPU acceleration via EGL
5. Output is encoded as GIF or MP4

## Technical Details

This implementation uses:
- ModernGL with EGL for GPU-accelerated rendering
- Multiple state-of-the-art depth estimation models
- BrokenSource's DepthFlow and ShaderFlow libraries
- NVIDIA GPU acceleration for rendering and depth estimation

## Depth Models

- **DepthAnything V2**: Latest Depth Anything model (fast, high quality)
- **DepthAnything V1**: Original Depth Anything model
- **DepthPro**: Specialized depth estimation model
- **Marigold**: Diffusion-based depth estimator
- **ZoeDepth**: High-quality depth with metric scaling

## Animation Presets

- **Orbital**: Camera rotating around the subject
- **Dolly**: Camera moving forward/backward
- **Zoom**: Smooth zoom effect
- **Focus**: Depth-based focus/defocus effect
- **Parallax**: Classic parallax motion effect

## References

- DepthFlow: https://github.com/BrokenSource/DepthFlow
- ShaderFlow: https://github.com/BrokenSource/ShaderFlow
- BrokenSource: https://github.com/BrokenSource/BrokenSource
- Replicate: https://replicate.com/docs/guides/push-a-model