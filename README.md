# DepthFlow on Replicate - Phase 2

This is an incremental implementation of the [DepthFlow](https://github.com/BrokenSource/DepthFlow) project on Replicate. This version incorporates the basic GLSL shader rendering with simplified depth map generation.

## Current Implementation

This version includes:

1. EGL context creation for headless GPU rendering
2. Basic GLSL shader for parallax depth effects
3. Simple depth map generation from image grayscale
4. Basic animation effects (orbital, zoom, horizontal)
5. GIF output generation

## Input Parameters

- `image`: Input image for applying the 3D effect
- `effect_type`: Type of 3D effect (choices: orbital, zoom, horizontal)
- `depth_strength`: Strength of the depth effect (default: 0.05, range: 0.01-0.2)
- `num_frames`: Number of frames to render (default: 30, range: 10-120)

## How It Works

1. The input image is processed to create a simple depth map based on image contrast
2. ModernGL with EGL backend is used to render frames using the depth map
3. A GLSL shader applies the parallax effect based on the depth information
4. Frames are combined into an animated GIF

## Technical Details

This implementation uses:
- ModernGL with EGL for GPU-accelerated rendering
- OpenCV for basic image processing
- PyTorch for CUDA verification (future depth models)
- GLSL shaders for the parallax effect

## Next Steps

The next phase will implement:
1. Advanced depth estimation models (DepthAnything, ZoeDepth, etc.)
2. Higher quality animation techniques
3. Video output with improved frame interpolation
4. Additional camera motion effects

## References

- Original DepthFlow repository: https://github.com/BrokenSource/DepthFlow
- Replicate documentation: https://replicate.com/docs/guides/push-a-model