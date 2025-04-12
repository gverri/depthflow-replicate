import os
import tempfile
import subprocess
import time
import traceback
import numpy as np
import moderngl
from cog import BasePredictor, Input, Path
# Import core DepthFlow modules
from DepthFlow.Scene import DepthScene
from DepthFlow.Animation import Animation, Target
from Broken.Externals.Depthmap import (
				DepthAnythingV1,
				DepthAnythingV2,
				DepthPro,
				Marigold,
				ZoeDepth,
)

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
        self.DepthAnythingV1 = DepthAnythingV1
        self.DepthPro = DepthPro
        self.Marigold = Marigold
        self.ZoeDepth = ZoeDepth

        self.has_all_models = True
        print("✅ DepthFlow imports successful with all depth models!")

    def predict(
        self,
        image: Path = Input(description="Input image for 3D effect"),
        depth_model: str = Input(
            description="Depth estimation model",
            choices=["anything2", "anything1", "depthpro", "marigold", "zoedepth"],
            default="anything2"
        ),
        animation_preset: str = Input(
            description="Animation preset",
            choices=["orbital", "dolly", "zoom", "focus", "parallax", "vertical", "horizontal", "circle"],
            default="orbital"
        ),
        depth_strength: float = Input(
            description="Strength of the depth effect",
            default=0.5,
            ge=0.1,
            le=1.0
        ),
        frames: int = Input(
            description="Number of frames to render",
            default=48,
            ge=24,
            le=120
        ),
        fps: int = Input(
            description="Frames per second",
            default=24,
            ge=12,
            le=60
        ),
        output_format: str = Input(
            description="Output format",
            choices=["mp4", "gif"],
            default="gif"
        ),
        isometric: float = Input(
            description="Isometric effect strength (0.0-1.0)",
            default=0.0,
            ge=0.0,
            le=1.0
        )
    ) -> Path:
        """Generate a 3D animation with depth effect using the DepthFlow library"""
        # Create temporary directory for output
        temp_dir = tempfile.mkdtemp()
        output_file = f"output.{output_format}"
        output_path = os.path.join(temp_dir, output_file)

        # If we made it here, we have at least the basic DepthFlow functionality
        print("Creating DepthScene...")
        # Create a DepthScene instance with headless backend for Replicate
        scene = self.DepthScene(backend="headless")

        # Select depth estimator based on user choice if available
        if self.has_all_models:
            print(f"Setting up depth estimator: {depth_model}")
            if depth_model == "anything2":
                scene.set_estimator(self.DepthAnythingV2())
            elif depth_model == "anything1":
                scene.set_estimator(self.DepthAnythingV1())
            elif depth_model == "depthpro":
                scene.set_estimator(self.DepthPro())
            elif depth_model == "marigold":
                scene.set_estimator(self.Marigold())
            elif depth_model == "zoedepth":
                scene.set_estimator(self.ZoeDepth())
        else:
            print("Using default depth estimator (models not fully loaded)")

        # Load the input image
        print(f"Loading input image: {image}")
        scene.input(image=str(image), depth=None)

        # Set up the animation preset
        print(f"Setting up animation preset: {animation_preset}")

        # Set depth strength
        scene.state.depth_strength = depth_strength

        # Set isometric effect if requested
        if isometric > 0:
            scene.set(
                target=self.Target.Isometric,
                value=isometric
            )

        # Add the animation preset
        if animation_preset == "orbital":
            scene.orbital(intensity=depth_strength)
        elif animation_preset == "dolly":
            scene.dolly(intensity=depth_strength)
        elif animation_preset == "zoom":
            scene.zoom(intensity=depth_strength)
        elif animation_preset == "focus":
            scene.focus()
        elif animation_preset == "parallax":
            scene.parallax(intensity=depth_strength)
        elif animation_preset == "vertical":
            scene.vertical(intensity=depth_strength)
        elif animation_preset == "horizontal":
            scene.horizontal(intensity=depth_strength)
        elif animation_preset == "circle":
            scene.circle(intensity=depth_strength)
        else:
            # Default to Orbital animation if none specified
            scene.orbital(intensity=depth_strength)

        # Set rendering parameters
        scene.fps = fps
        scene.frames = frames

        # Run the animation rendering
        print(f"Rendering {frames} frames with {animation_preset} effect...")
        start_time = time.time()
        scene.main(output=output_path)

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