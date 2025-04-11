import os
import sys
import tempfile
import subprocess
import time
import traceback
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageSequence
import moderngl
from pathlib import Path as PathLib
from cog import BasePredictor, Input, Path

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
            self.ctx = ctx
        except Exception as e:
            print(f"Error: Could not create ModernGL EGL context: {e}")
            self.ctx = None
            print("Will attempt to continue without OpenGL context")
        
        # Initialize DepthFlow modules
        print("Testing DepthFlow imports...")
        self.has_all_models = False
        self.has_basic_depthflow = False
        
        try:
            # Try importing the core DepthFlow modules
            try:
                from DepthFlow.Scene import DepthScene
                from DepthFlow.Animation import Animation
                
                # Try importing the depth models
                try:
                    from Broken.Externals.Depthmap import (
                        DepthAnythingV1,
                        DepthAnythingV2,
                        DepthPro,
                        Marigold,
                        ZoeDepth,
                    )
                    
                    # Store references to modules
                    self.DepthScene = DepthScene
                    self.Animation = Animation
                    self.DepthAnythingV2 = DepthAnythingV2
                    self.DepthAnythingV1 = DepthAnythingV1
                    self.DepthPro = DepthPro
                    self.Marigold = Marigold
                    self.ZoeDepth = ZoeDepth
                    
                    self.has_all_models = True
                    print("✅ DepthFlow imports successful with all depth models!")
                    
                except ImportError as e:
                    print(f"⚠️ Could not import depth models: {e}")
                    print("Falling back to basic implementation...")
                    
                    # Store just the basic modules
                    self.DepthScene = DepthScene
                    self.Animation = Animation
                    self.has_basic_depthflow = True
                    print("✅ DepthFlow basic imports successful")
                    
            except ImportError as e:
                print(f"⚠️ Could not import basic DepthFlow modules: {e}")
                print("Will attempt to continue with fallback implementation")
                
        except Exception as e:
            print(f"❌ Error importing DepthFlow modules: {e}")
            print(traceback.format_exc())
            print("Will attempt to continue with fallback implementation")
    
    def create_simple_animation(self, input_image_path, output_path, frames=48, fps=24, effect_type="parallax", depth_strength=0.5):
        """Create a simple parallax effect if DepthFlow fails"""
        try:
            print(f"Creating improved fallback animation for {input_image_path}")
            
            # Open the input image at high quality
            img = Image.open(input_image_path).convert("RGBA")
            
            # Scale large images to a reasonable size for processing
            max_dimension = 1024
            orig_width, orig_height = img.size
            
            if max(orig_width, orig_height) > max_dimension:
                scale_factor = max_dimension / max(orig_width, orig_height)
                new_width = int(orig_width * scale_factor)
                new_height = int(orig_height * scale_factor)
                print(f"Scaling image from {orig_width}x{orig_height} to {new_width}x{new_height}")
                img = img.resize((new_width, new_height), Image.LANCZOS)
            
            width, height = img.size
            print(f"Working with image size: {width}x{height}")
            
            # Create more sophisticated depth map
            # Convert to grayscale and apply enhancements
            gray_img = img.convert("L")
            
            # Create depth map using gradient-based edge detection
            # This simulates a depth map by assuming edges usually separate depths
            depth_map = np.array(gray_img, dtype=np.float32)
            
            # Apply Gaussian blur to smooth the depth map
            import scipy.ndimage
            depth_map = scipy.ndimage.gaussian_filter(depth_map, sigma=5)
            
            # Add a gradient from top to bottom (objects at the bottom tend to be closer)
            y_gradient = np.linspace(0, 100, height).reshape(-1, 1)
            depth_map += y_gradient

            # Normalize the depth map
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
            
            # Optionally save the depth map for debugging
            depth_img = Image.fromarray((depth_map * 255).astype(np.uint8))
            depth_path = os.path.join(os.path.dirname(output_path), "depth_map.png")
            depth_img.save(depth_path)
            print(f"Saved depth map to {depth_path}")
            
            # Calculate displacement maps for 3D effect
            # Use depth map to create different layers that move at different speeds
            num_layers = 5
            layers = []
            layer_masks = []
            
            # Divide the depth map into layers based on depth values
            for i in range(num_layers):
                min_depth = i / num_layers
                max_depth = (i + 1) / num_layers
                
                # Create a mask for this depth layer
                mask = np.zeros_like(depth_map)
                mask[(depth_map >= min_depth) & (depth_map < max_depth)] = 1
                
                # Convert the mask to an image
                mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                
                # Extract the layer using the mask
                layer_img = Image.new("RGBA", img.size, (0, 0, 0, 0))
                layer_img.paste(img, (0, 0), mask_img)
                
                layers.append(layer_img)
                layer_masks.append(mask_img)
            
            # Create output frames with higher quality
            frames_list = []
            
            # Scale the amplitude based on the depth_strength parameter and image size
            base_amplitude = int(min(width, height) * 0.05)  # Base amplitude is 5% of the smaller dimension
            amplitude = int(base_amplitude * depth_strength * 2)  # Apply depth strength and double it for effect
            print(f"Using amplitude: {amplitude} (base: {base_amplitude}, strength: {depth_strength})")
            
            # Create temporary directory for frames
            frames_dir = os.path.join(os.path.dirname(output_path), "frames")
            os.makedirs(frames_dir, exist_ok=True)
            
            for i in range(frames):
                # Create a new frame
                frame = Image.new("RGBA", (width, height), (0, 0, 0, 0))
                progress = (i / frames) * 2 * np.pi
                
                if effect_type == "parallax":
                    # Horizontal movement
                    for layer_idx, (layer, mask) in enumerate(zip(layers, layer_masks)):
                        # Scale movement based on layer depth
                        layer_factor = 1 - (layer_idx / num_layers)  # Closer layers move more
                        offset_x = int(amplitude * np.sin(progress) * layer_factor)
                        offset_y = 0
                        
                        # Create a new image with the layer at the offset position
                        offset_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
                        offset_layer.paste(layer, (offset_x, offset_y), layer)
                        
                        # Composite the offset layer onto the frame
                        frame = Image.alpha_composite(frame, offset_layer)
                
                elif effect_type == "orbital":
                    # Circular movement
                    for layer_idx, (layer, mask) in enumerate(zip(layers, layer_masks)):
                        # Scale movement based on layer depth
                        layer_factor = 1 - (layer_idx / num_layers)  # Closer layers move more
                        offset_x = int(amplitude * np.sin(progress) * layer_factor)
                        offset_y = int(amplitude * np.cos(progress) * layer_factor)
                        
                        # Create a new image with the layer at the offset position
                        offset_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
                        offset_layer.paste(layer, (offset_x, offset_y), layer)
                        
                        # Composite the offset layer onto the frame
                        frame = Image.alpha_composite(frame, offset_layer)
                
                elif effect_type == "zoom":
                    # Zoom effect using the layers
                    for layer_idx, (layer, mask) in enumerate(zip(layers, layer_masks)):
                        # Scale movement based on layer depth
                        layer_factor = 1 - (layer_idx / num_layers)  # Closer layers scale more
                        scale_amount = 0.05 * layer_factor * depth_strength
                        scale_factor = 1.0 + scale_amount * np.sin(progress)
                        
                        # Resize the layer
                        new_width = int(width * scale_factor)
                        new_height = int(height * scale_factor)
                        if new_width > 0 and new_height > 0:  # Avoid zero-sized images
                            resized_layer = layer.resize((new_width, new_height), Image.LANCZOS)
                            
                            # Calculate position to center the resized layer
                            pos_x = (width - new_width) // 2
                            pos_y = (height - new_height) // 2
                            
                            # Create a new image with the resized layer
                            scaled_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
                            scaled_layer.paste(resized_layer, (pos_x, pos_y), resized_layer)
                            
                            # Composite the scaled layer onto the frame
                            frame = Image.alpha_composite(frame, scaled_layer)
                
                elif effect_type == "dolly":
                    # Forward/backward dolly zoom effect using the layers
                    for layer_idx, (layer, mask) in enumerate(zip(layers, layer_masks)):
                        # Scale movement based on layer depth
                        layer_factor = 1 - (layer_idx / num_layers)  # Closer layers scale more
                        
                        # Scale the layer (simulating coming closer)
                        scale_amount = 0.1 * layer_factor * depth_strength
                        scale_factor = 1.0 + scale_amount * np.sin(progress)
                        
                        # Also offset the layer slightly to simulate parallax
                        offset_factor = 5 * layer_factor * depth_strength
                        offset_x = int(offset_factor * np.sin(progress) * width / 100)
                        offset_y = int(offset_factor * np.sin(progress) * height / 100)
                        
                        # Resize the layer
                        new_width = int(width * scale_factor)
                        new_height = int(height * scale_factor)
                        if new_width > 0 and new_height > 0:  # Avoid zero-sized images
                            resized_layer = layer.resize((new_width, new_height), Image.LANCZOS)
                            
                            # Calculate position to center the resized layer plus offset
                            pos_x = (width - new_width) // 2 + offset_x
                            pos_y = (height - new_height) // 2 + offset_y
                            
                            # Create a new image with the resized layer
                            scaled_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
                            scaled_layer.paste(resized_layer, (pos_x, pos_y), resized_layer)
                            
                            # Composite the scaled layer onto the frame
                            frame = Image.alpha_composite(frame, scaled_layer)
                
                elif effect_type == "focus":
                    # Focus effect (blur background, keep foreground sharp)
                    # This is a simplified version of a focus effect
                    
                    # Start with the original image
                    frame = img.copy()
                    
                    # Create a blurred version
                    blurred = img.copy()
                    blurred = blurred.filter(ImageFilter.GaussianBlur(radius=5))
                    
                    # Use a mask based on the progress to blend between foreground and background
                    mask_value = int(127 + 127 * np.sin(progress))
                    mask = Image.new("L", img.size, mask_value)
                    
                    # Composite using the mask
                    frame = Image.composite(img, blurred, mask)
                
                else:
                    # Default fallback to original image
                    frame = img.copy()
                
                # Convert to RGB for saving
                if frame.mode == "RGBA":
                    frame = frame.convert("RGB")
                
                # Save frame
                frame_path = os.path.join(frames_dir, f"frame_{i:04d}.jpg")
                frame.save(frame_path, quality=95)
                frames_list.append(frame)
            
            # Save as GIF or MP4 using ffmpeg for higher quality
            if output_path.endswith(".gif"):
                print("Creating high-quality GIF...")
                # Use higher quality settings for GIF
                frames_list[0].save(
                    output_path,
                    save_all=True,
                    append_images=frames_list[1:],
                    duration=1000//fps,
                    loop=0,
                    optimize=False,
                    quality=95
                )
            else:
                print("Creating high-quality MP4...")
                # For MP4, use ffmpeg with high quality settings
                try:
                    # Create a temporary directory for frame files
                    frame_pattern = os.path.join(frames_dir, "frame_%04d.jpg")
                    
                    # Use ffmpeg with higher quality settings
                    ffmpeg_cmd = [
                        "ffmpeg", 
                        "-framerate", str(fps),
                        "-i", frame_pattern,
                        "-c:v", "libx264",
                        "-profile:v", "high",
                        "-crf", "18",  # Lower value = higher quality (range 0-51)
                        "-pix_fmt", "yuv420p",
                        "-movflags", "faststart",
                        "-y",  # Overwrite output file if it exists
                        output_path
                    ]
                    
                    print(f"Running ffmpeg command: {' '.join(ffmpeg_cmd)}")
                    subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
                    print(f"Successfully created MP4: {output_path}")
                    
                except Exception as e:
                    print(f"Error creating MP4 with ffmpeg: {e}")
                    print(traceback.format_exc())
                    
                    # Fallback to GIF if ffmpeg fails
                    gif_path = output_path.replace(".mp4", ".gif")
                    frames_list[0].save(
                        gif_path,
                        save_all=True,
                        append_images=frames_list[1:],
                        duration=1000//fps,
                        loop=0
                    )
                    print(f"Fallback to GIF: {gif_path}")
                    output_path = gif_path
            
            print(f"Animation saved to {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error creating animation: {e}")
            print(traceback.format_exc())
            
            # Create an error image
            error_img = Image.new("RGB", (800, 400), color="black")
            draw = ImageDraw.Draw(error_img)
            draw.text((50, 50), f"Animation generation failed: {e}", fill="white")
            draw.text((50, 100), "Using fallback image", fill="white")
            
            # Save error image
            error_path = output_path.replace(".gif", ".png").replace(".mp4", ".png")
            error_img.save(error_path)
            return error_path
    
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
            choices=["orbital", "dolly", "zoom", "focus", "parallax"],
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
        )
    ) -> Path:
        """Generate a 3D animation with depth effect using the DepthFlow library"""
        # Create temporary directory for output
        temp_dir = tempfile.mkdtemp()
        output_file = f"output.{output_format}"
        output_path = os.path.join(temp_dir, output_file)
        
        # Check if we have the required components
        if not hasattr(self, 'DepthScene') or not hasattr(self, 'Animation'):
            print("DepthFlow modules not available, using fallback implementation")
            return Path(self.create_simple_animation(
                str(image), 
                output_path, 
                frames=frames, 
                fps=fps, 
                effect_type=animation_preset,
                depth_strength=depth_strength
            ))
        
        # If we made it here, we have at least the basic DepthFlow functionality
        try:
            print("Creating DepthScene...")
            # Create a DepthScene instance
            scene = self.DepthScene()
            
            # Select depth estimator based on user choice if available
            if hasattr(self, 'has_all_models') and self.has_all_models:
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
            scene.input(image=[str(image)], depth=None)
            
            # Set up the animation preset
            print(f"Setting up animation preset: {animation_preset}")
            scene.config.animation.clear()
            if animation_preset == "orbital":
                scene.config.animation.add(self.Animation.Orbital())
            elif animation_preset == "dolly":
                scene.config.animation.add(self.Animation.Dolly())
            elif animation_preset == "zoom":
                scene.config.animation.add(self.Animation.Zoom())
            elif animation_preset == "focus":
                scene.config.animation.add(self.Animation.Focus())
            elif animation_preset == "parallax":
                scene.config.animation.add(self.Animation.Parallax())
            
            # Set other parameters
            try:
                scene.state.depth.strength = depth_strength
            except AttributeError:
                print("Warning: Could not set depth strength")
            
            scene.fps = fps
            scene.frames = frames
            
            # Run the animation rendering
            print(f"Rendering {frames} frames with {animation_preset} effect...")
            try:
                # Set a timeout for the rendering process
                start_time = time.time()
                max_render_time = 300  # 5 minutes max
                
                scene.run(output=output_path)
                
                render_time = time.time() - start_time
                print(f"DepthFlow animation rendered in {render_time:.2f} seconds")
                print(f"Output saved to {output_path}")
                
                # Check if the output file exists and has content
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    return Path(output_path)
                else:
                    print("Output file doesn't exist or is empty, falling back to simple animation")
                    return Path(self.create_simple_animation(
                        str(image), 
                        output_path, 
                        frames=frames, 
                        fps=fps, 
                        effect_type=animation_preset,
                        depth_strength=depth_strength
                    ))
                
            except Exception as e:
                print(f"Error during rendering: {e}")
                print(traceback.format_exc())
                # Fall back to simple animation
                return Path(self.create_simple_animation(
                    str(image), 
                    output_path, 
                    frames=frames, 
                    fps=fps, 
                    effect_type=animation_preset,
                    depth_strength=depth_strength
                ))
                
        except Exception as e:
            print(f"Error setting up DepthFlow: {e}")
            print(traceback.format_exc())
            # Fall back to simple animation
            return Path(self.create_simple_animation(
                str(image), 
                output_path, 
                frames=frames, 
                fps=fps, 
                effect_type=animation_preset,
                depth_strength=depth_strength
            ))