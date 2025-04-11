import os
import sys
import time
import tempfile
import subprocess
import moderngl
import numpy as np
from PIL import Image
import cv2
import torch
from cog import BasePredictor, Input, Path

# Shader for depth effect
DEPTH_SHADER = """
    #version 330

    uniform sampler2D image;
    uniform sampler2D depth;
    uniform float time;
    uniform float depth_strength;
    uniform int effect_type;
    
    in vec2 uv;
    out vec4 fragColor;
    
    vec2 parallax(vec2 uv, float strength) {
        float depth_value = texture(depth, uv).r;
        vec2 direction = vec2(0.0, 0.0);
        
        // Different movement patterns based on effect type
        if (effect_type == 0) {
            // Orbital
            direction = vec2(cos(time * 2.0), sin(time * 2.0));
        } else if (effect_type == 1) {
            // Zoom
            direction = (uv - 0.5) * 2.0;
        } else if (effect_type == 2) {
            // Horizontal
            direction = vec2(sin(time * 2.0), 0.0);
        }
        
        return uv - direction * depth_value * strength;
    }
    
    void main() {
        vec2 shifted_uv = parallax(uv, depth_strength);
        
        // Sample the image with the shifted UV coordinates
        if (shifted_uv.x < 0.0 || shifted_uv.x > 1.0 || 
            shifted_uv.y < 0.0 || shifted_uv.y > 1.0) {
            fragColor = vec4(0.0, 0.0, 0.0, 1.0);
        } else {
            fragColor = texture(image, shifted_uv);
        }
    }
"""

class Predictor(BasePredictor):
    def setup(self):
        """Set up ModernGL with EGL for headless rendering"""
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
            self.ctx = moderngl.create_standalone_context(backend="egl")
            print(f"Successfully created ModernGL context with EGL:")
            print(f"  GL_RENDERER: {self.ctx.info['GL_RENDERER']}")
            print(f"  GL_VERSION: {self.ctx.info['GL_VERSION']}")
        except Exception as e:
            print(f"Error: Could not create ModernGL EGL context: {e}")
            sys.exit(1)
        
        # Create the shader program
        self.prog = self.ctx.program(
            vertex_shader="""
                #version 330
                
                in vec2 in_position;
                in vec2 in_texcoord;
                
                out vec2 uv;
                
                void main() {
                    gl_Position = vec4(in_position, 0.0, 1.0);
                    uv = in_texcoord;
                }
            """,
            fragment_shader=DEPTH_SHADER
        )
        
        # Create a full-screen quad
        vertices = np.array([
            # x      y     u     v
            -1.0,  1.0,  0.0,  0.0,  # top left
            -1.0, -1.0,  0.0,  1.0,  # bottom left
             1.0, -1.0,  1.0,  1.0,  # bottom right
            -1.0,  1.0,  0.0,  0.0,  # top left
             1.0, -1.0,  1.0,  1.0,  # bottom right
             1.0,  1.0,  1.0,  0.0,  # top right
        ], dtype='f4')
        
        self.vbo = self.ctx.buffer(vertices)
        self.vao = self.ctx.vertex_array(
            self.prog,
            [
                (self.vbo, '2f 2f', 'in_position', 'in_texcoord'),
            ],
        )
        
        # Try to import MiDaS depth model (minimal)
        try:
            import torch
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            if torch.cuda.is_available():
                print(f"PyTorch will use CUDA: {torch.cuda.get_device_name(0)}")
            else:
                print("PyTorch will use CPU (CUDA not available)")
        except Exception as e:
            print(f"PyTorch import warning (depth estimation will use grayscale): {e}")
    
    def generate_depth_from_grayscale(self, image_path):
        """Generate a simple depth map from image grayscale (for testing)"""
        # Load the image and convert to grayscale
        img = cv2.imread(str(image_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Edge detection for rough depth approximation
        edges = cv2.Canny(gray, 100, 200)
        
        # Apply Gaussian blur to smooth the edges
        depth = cv2.GaussianBlur(255 - gray, (15, 15), 0)
        depth = depth / 255.0  # Normalize to 0-1
        
        return Image.fromarray((depth * 255).astype(np.uint8), 'L')
    
    def render_frame(self, image, depth, time_value, effect_type, depth_strength):
        """Render a single frame with the depth effect"""
        # Upload textures
        image_texture = self.ctx.texture(image.size, 4, image.tobytes())
        depth_texture = self.ctx.texture(depth.size, 1, depth.tobytes())
        
        # Create framebuffer
        fbo = self.ctx.framebuffer(color_attachments=[self.ctx.texture(image.size, 4)])
        fbo.use()
        
        # Set uniforms
        image_texture.use(0)
        depth_texture.use(1)
        self.prog['image'] = 0
        self.prog['depth'] = 1
        self.prog['time'] = time_value
        self.prog['depth_strength'] = depth_strength
        self.prog['effect_type'] = effect_type
        
        # Render
        self.ctx.clear()
        self.vao.render()
        
        # Read the result
        data = fbo.read(components=4)
        result = Image.frombytes('RGBA', image.size, data)
        
        # Clean up
        fbo.release()
        image_texture.release()
        depth_texture.release()
        
        return result
    
    def predict(
        self,
        image: Path = Input(description="Input image for 3D effect"),
        effect_type: str = Input(
            description="Type of 3D effect", 
            choices=["orbital", "zoom", "horizontal"],
            default="orbital"
        ),
        depth_strength: float = Input(
            description="Strength of the depth effect",
            default=0.05,
            ge=0.01,
            le=0.2
        ),
        num_frames: int = Input(
            description="Number of frames to render",
            default=30,
            ge=10,
            le=120
        )
    ) -> Path:
        """Create a simple GIF with depth effect"""
        # Determine effect type index
        effect_map = {"orbital": 0, "zoom": 1, "horizontal": 2}
        effect_index = effect_map[effect_type]
        
        # Load input image
        image_pil = Image.open(str(image)).convert('RGBA')
        
        # Resize image if too large (for performance)
        max_size = 1024
        if image_pil.width > max_size or image_pil.height > max_size:
            ratio = min(max_size / image_pil.width, max_size / image_pil.height)
            new_width = int(image_pil.width * ratio)
            new_height = int(image_pil.height * ratio)
            image_pil = image_pil.resize((new_width, new_height), Image.LANCZOS)
        
        # Generate depth map (just a grayscale conversion for now)
        print("Generating depth map...")
        depth_map = self.generate_depth_from_grayscale(image)
        
        # Ensure depth map is the same size as the image
        if depth_map.size != image_pil.size:
            depth_map = depth_map.resize(image_pil.size, Image.LANCZOS)
        
        # Create temporary directory for output
        temp_dir = tempfile.mkdtemp()
        output_path = Path(os.path.join(temp_dir, "output.gif"))
        
        # Render frames
        print(f"Rendering {num_frames} frames...")
        frames = []
        for i in range(num_frames):
            time_value = i / num_frames * 2 * np.pi  # 0 to 2π
            frame = self.render_frame(
                image_pil, depth_map, time_value, effect_index, depth_strength
            )
            frames.append(frame)
        
        # Save as GIF
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            optimize=False,
            duration=1000 // 15,  # ~15 fps
            loop=0
        )
        
        print(f"Saved output GIF to {output_path}")
        return output_path