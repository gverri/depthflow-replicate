# DepthFlow on Replicate

This is an implementation of the [DepthFlow](https://github.com/BrokenSource/DepthFlow) project on Replicate, providing 3D parallax animations from static images.

## Features

This implementation includes:

1. EGL context creation for headless GPU rendering
2. State-of-the-art depth estimation using DepthAnythingV2
3. Various animation presets (orbital, dolly, zoom, focus, parallax, vertical, horizontal, circle)
4. Output in MP4 format
5. Configurable animation parameters for depth effects and post-processing

## Input Parameters

- `image`: Input image for applying the 3D effect
- `animation_preset`: Animation style (choices: orbital, dolly, zoom, focus, parallax, vertical, horizontal, circle)
- `loop_animation`: Loop the animation instead of once-through (default: true)
- `depth_strength`: Strength of the depth effect (default: 0.5, range: 0.1-1.0)
- `fps`: Frames per second (default: 24, range: 12-60)
- `duration`: Video duration in seconds (default: 2.0, range: 1.0-10.0)
- `isometric`: Isometric effect strength (default: 0.0, range: 0.0-1.0)
- `steady`: Depth plane anchor point (0.0=background fixed, 1.0=foreground fixed) (default: 0.3)
- `zoom`: Camera zoom factor (default: 1.0, range: 0.5-1.5)
- `vignette`: Apply vignette effect (default: false)
- `blur`: Apply depth of field blur effect (default: false)
- `blur_intensity`: Blur effect intensity if blur is enabled (default: 1.0, range: 0.1-2.0)
- `color_enhance`: Apply color enhancement (default: false)
- `saturation`: Image saturation adjustment if color enhance is enabled (default: 1.2, range: 0.5-2.0)
- `contrast`: Image contrast adjustment if color enhance is enabled (default: 1.1, range: 0.5-2.0)

## How It Works

1. The input image is processed by the DepthAnythingV2 depth estimation model
2. The DepthFlow library creates a 3D scene from the image and depth map
3. The selected animation preset is applied to the scene
4. Frames are rendered using GPU acceleration via EGL
5. Output is encoded as MP4

## Technical Details

This implementation uses:
- ModernGL with EGL for GPU-accelerated rendering
- DepthAnythingV2 for high-quality depth estimation
- BrokenSource's DepthFlow and ShaderFlow libraries
- NVIDIA GPU acceleration for rendering and depth estimation

## Animation Presets

- **Orbital**: Camera rotating around the subject
- **Dolly**: Camera moving forward/backward
- **Zoom**: Smooth zoom effect
- **Focus**: Depth-based focus/defocus effect
- **Parallax**: Classic parallax motion effect
- **Vertical**: Vertical camera movement
- **Horizontal**: Horizontal camera movement
- **Circle**: Circular camera movement

## References

- DepthFlow: https://github.com/BrokenSource/DepthFlow
- ShaderFlow: https://github.com/BrokenSource/ShaderFlow
- BrokenSource: https://github.com/BrokenSource/BrokenSource
- Replicate: https://replicate.com/docs/guides/push-a-model