import os
import tempfile
import subprocess
import time
import math
import numpy as np
import moderngl
from cog import BasePredictor, Input, Path
# Import core DepthFlow modules
from DepthFlow.Scene import DepthScene
from DepthFlow.Animation import Animation, Target
from Broken.Externals.Depthmap import ( DepthAnythingV2 )

class Predictor(BasePredictor):
    def setup(self):
        """Set up the DepthFlow environment and load models"""
        # Set environment variables for OpenGL
        os.environ["NVIDIA_VISIBLE_DEVICES"] = "all"
        os.environ["NVIDIA_DRIVER_CAPABILITIES"] = "all"

        # Configure EGL
        vendor_file = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
        if os.path.exists(vendor_file):
            print(f"EGL vendor configuration found at {vendor_file}")
        else:
            print(f"Warning: EGL vendor configuration not found at {vendor_file}")

        # Test NVIDIA GPU
        subprocess.run(["nvidia-smi"], check=True, capture_output=True)
        print("NVIDIA GPU detected.")

        # Test ModernGL context
        print("Testing EGL context creation...")
        ctx = moderngl.create_standalone_context(backend="egl")
        print(f"Successfully created ModernGL context with EGL:")
        print(f"  GL_RENDERER: {ctx.info['GL_RENDERER']}")
        print(f"  GL_VERSION: {ctx.info['GL_VERSION']}")
        self.ctx = ctx

        # Initialize DepthFlow modules
        print("Testing DepthFlow imports...")
        self.has_all_models = False

        # Store references to modules
        self.DepthScene = DepthScene
        self.Animation = Animation
        self.Target = Target
        self.DepthAnythingV2 = DepthAnythingV2
        self.has_all_models = True
        print("✅ DepthFlow imports successful with all depth models!")

    def predict(
        self,
        image: Path = Input(description="Input image for 3D effect"),
        animation_preset: str = Input(
            description="Animation preset",
            choices=["orbital", "dolly", "zoom", "focus", "parallax", "vertical", "horizontal", "circle"],
            default="orbital"
        ),
        loop_animation: bool = Input(
            description="Loop the animation instead of once-through (affects most animation presets)",
            default=True
        ),
        depth_strength: float = Input(
            description="Strength of the depth effect",
            default=0.5,
            ge=0.1,
            le=1.0
        ),
        fps: int = Input(
            description="Frames per second",
            default=24,
            ge=12,
            le=60
        ),
        duration: float = Input(
            description="Video duration in seconds",
            default=2.0,
            ge=1.0,
            le=10.0
        ),
        isometric: float = Input(
            description="Isometric effect strength (0.0-1.0)",
            default=0.0,
            ge=0.0,
            le=1.0
        ),
        steady: float = Input(
            description="Depth plane anchor point (0.0=background fixed, 1.0=foreground fixed)",
            default=0.3,
            ge=0.0,
            le=1.0
        ),
        zoom: float = Input(
            description="Camera zoom factor",
            default=1.0,
            ge=0.5,
            le=1.5
        ),
        vignette: bool = Input(
            description="Apply vignette effect",
            default=False
        ),
        blur: bool = Input(
            description="Apply depth of field blur effect",
            default=False
        ),
        blur_intensity: float = Input(
            description="Blur effect intensity (if blur is enabled)",
            default=1.0,
            ge=0.1,
            le=2.0
        ),
        color_enhance: bool = Input(
            description="Apply color enhancement",
            default=False
        ),
        saturation: float = Input(
            description="Image saturation adjustment (if color enhance is enabled)",
            default=1.2,
            ge=0.5,
            le=2.0
        ),
        contrast: float = Input(
            description="Image contrast adjustment (if color enhance is enabled)",
            default=1.1,
            ge=0.5,
            le=2.0
        )
    ) -> Path:
        """Generate a 3D animation with depth effect using the DepthFlow library"""
        # Create temporary directory for output
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, "output.mp4")

        # If we made it here, we have at least the basic DepthFlow functionality
        print("Creating DepthScene...")
        # Create a DepthScene instance with headless backend for Replicate
        scene = self.DepthScene(backend="headless")

        # Always use DepthAnythingV2 for depth estimation
        print("Setting up depth estimator: DepthAnythingV2")
        try:
            scene.set_estimator(self.DepthAnythingV2())
        except Exception as e:
            print(f"Error setting up depth estimator: {str(e)}")
            print("Will use the default depth estimator")

        # Load the input image
        print(f"Loading input image: {image}")
        scene.input(image=str(image), depth=None)

        # Configure basic scene parameters
        scene.state.isometric = isometric
        scene.state.steady = steady
        scene.state.zoom = zoom

        # Set up the animation preset
        print(f"Setting up animation preset: {animation_preset}")

        # Apply the animation preset with adjusted depth_strength and loop setting
        if animation_preset == "orbital":
            scene.orbital(intensity=depth_strength)
        elif animation_preset == "dolly":
            scene.dolly(intensity=depth_strength, focus=steady, loop=loop_animation)
        elif animation_preset == "zoom":
            scene.zoom(intensity=depth_strength, loop=loop_animation)
        elif animation_preset == "focus":
            # Set focus point using Set animation
            scene.set(target=self.Target.Focus, value=steady)
            # Add some gentle movement, focus doesn't have a loop option
            scene.config.animation.add(self.Animation.Sine(
                target=self.Target.Isometric,
                amplitude=0.3 * depth_strength,
                bias=0.5 * depth_strength,
                cycles=1.0 if loop_animation else 0.5
            ))
        elif animation_preset == "parallax":
            # Use horizontal motion for parallax effect
            scene.horizontal(intensity=depth_strength, steady=steady, loop=loop_animation)
        elif animation_preset == "vertical":
            scene.vertical(intensity=depth_strength, steady=steady, loop=loop_animation)
        elif animation_preset == "horizontal":
            scene.horizontal(intensity=depth_strength, steady=steady, loop=loop_animation)
        elif animation_preset == "circle":
            # Circle doesn't have a loop option in the original implementation
            scene.circle(intensity=depth_strength)
        else:
            # Default to Orbital animation if none specified
            scene.orbital(intensity=depth_strength)

        # Add post-processing effects if requested
        if vignette:
            scene.vignette(intensity=0.2, decay=20)

        if blur:
            scene.blur(
                intensity=blur_intensity,
                start=0.6,
                end=1.0,
                exponent=2.0,
                quality=4,
                directions=16
            )

        if color_enhance:
            scene.colors(
                saturation=saturation * 100,  # Convert to percentage
                contrast=contrast * 100,      # Convert to percentage
                brightness=100,               # Default
                gamma=100                     # Default
            )

        # Log the expected configuration
        expected_frames = math.ceil(duration * fps)
        print(f"Animation configuration:")
        print(f"- Target Duration: {duration}s")
        print(f"- Target FPS: {fps}")
        print(f"- Expected Frames: {expected_frames}")
        print(f"Note: Actual duration may vary slightly due to video container/codec constraints")

        # Run the animation rendering using the author's recommended approach
        print(f"Rendering animation with {animation_preset} effect...")
        start_time = time.time()

        # Use the author's recommended way to control duration and fps
        scene.main(time=duration, fps=fps, output=output_path)

        render_time = time.time() - start_time
        print(f"DepthFlow animation rendered in {render_time:.2f} seconds")
        print(f"Output saved to {output_path}")

        # Clean up OpenGL resources
        if scene.window is not None:
            scene.window.destroy()

        # Check if the output file exists and has content
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return Path(output_path)
        else:
            raise RuntimeError("Failed to generate output file - file is empty or doesn't exist")