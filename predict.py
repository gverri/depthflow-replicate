import os
import subprocess
import sys
import moderngl
from cog import BasePredictor, Input, Path

class Predictor(BasePredictor):
    def setup(self):
        """Test if ModernGL with EGL works"""
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
        try:
            subprocess.run(["nvidia-smi"], check=True, capture_output=True)
            print("NVIDIA GPU detected.")
        except subprocess.CalledProcessError:
            print("NVIDIA GPU not detected!")
            
        # Test ModernGL context
        print("Testing EGL context creation...")
        try:
            ctx = moderngl.create_standalone_context(backend="egl")
            print(f"Successfully created ModernGL context with EGL:")
            print(f"  GL_RENDERER: {ctx.info['GL_RENDERER']}")
            print(f"  GL_VERSION: {ctx.info['GL_VERSION']}")
            print("All context info:")
            for key, value in ctx.info.items():
                print(f"  {key}: {value}")
        except Exception as e:
            print(f"Error: Could not create ModernGL EGL context: {e}")
            try:
                print("Trying alternative method...")
                ctx = moderngl.create_context(standalone=True)
                print(f"Alternative method worked! GL_RENDERER: {ctx.info['GL_RENDERER']}")
            except Exception as e2:
                print(f"Alternative method also failed: {e2}")

    def predict(self, image: Path = Input(description="Dummy input (not used)")) -> Path:
        """Dummy predict function"""
        # Just return the input as is
        return image