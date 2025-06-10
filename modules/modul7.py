"""
Modul 7: Pengolahan Citra Digital - Color Space Conversion
Implementasi konversi berbagai model warna: RGB, XYZ, LAB, YCrCb, YUV, YIQ, HSI, HSV, CIELuv
Compatible dengan FastAPI Framework
Author: RINDI INDRIANI - 231511030
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from fastapi import UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import os
import io
import base64
from PIL import Image
import math

class Modul7:
    """
    Class untuk Modul 7 - Color Space Conversion
    Compatible dengan FastAPI framework
    """
    
    def __init__(self, templates: Jinja2Templates):
        self.templates = templates
        self.dataset_path = "static/dataset/"
        self.uploads_path = "static/uploads/"
        self.results_path = "static/color_results/"
          # Ensure results directory exists
        os.makedirs(self.results_path, exist_ok=True)
    
    async def home(self, request):
        """Home page untuk Modul 7"""
        return self.templates.TemplateResponse("modules/modul7/home.html", {
            "request": request,
            "title": "Modul 7: Color Space Conversion"
        })
    
    def save_plot_as_base64(self, fig):
        """Convert matplotlib figure to base64 string"""
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        plot_data = buffer.getvalue()
        buffer.close()
        plt.close(fig)
        
        plot_url = base64.b64encode(plot_data).decode()
        return f"data:image/png;base64,{plot_url}"
    
    def normalize_for_display(self, img):
        """Normalize image for proper display"""
        if img is None:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Handle different image formats
        if len(img.shape) == 2:
            # Grayscale image
            img_normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            return np.stack([img_normalized] * 3, axis=-1).astype(np.uint8)
        elif len(img.shape) == 3:
            # Color image
            img_normalized = np.zeros_like(img, dtype=np.uint8)
            for i in range(img.shape[2]):
                channel = img[:,:,i]
                if np.isnan(channel).any():
                    channel = np.nan_to_num(channel, nan=0.0)
                img_normalized[:,:,i] = cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX)
            return img_normalized
        else:
            return np.zeros((100, 100, 3), dtype=np.uint8)
    
    def rgb_to_yiq_manual(self, rgb):
        """Manual implementation of RGB to YIQ conversion"""
        # YIQ transformation matrix (NTSC standard)
        transform_matrix = np.array([
            [0.299, 0.587, 0.114],
            [0.59590059, -0.27455667, -0.32134392],
            [0.21153661, -0.52273617, 0.31119955]
        ])
        
        # Normalize RGB to 0-1 range
        rgb_normalized = rgb.astype(np.float32) / 255.0
        
        # Reshape for matrix multiplication
        original_shape = rgb_normalized.shape
        rgb_flat = rgb_normalized.reshape(-1, 3)
        
        # Apply transformation
        yiq_flat = np.dot(rgb_flat, transform_matrix.T)
        
        # Reshape back
        yiq = yiq_flat.reshape(original_shape)
        
        # Normalize for display
        yiq_normalized = np.zeros_like(yiq)
        for i in range(3):
            channel = yiq[:, :, i]
            yiq_normalized[:, :, i] = (channel - np.min(channel)) / (np.max(channel) - np.min(channel))
        
        return (yiq_normalized * 255).astype(np.uint8)
    
    def rgb_to_hsi_manual(self, rgb):
        """Manual implementation of RGB to HSI conversion"""
        # Handle division by zero and invalid operations
        with np.errstate(divide='ignore', invalid='ignore'):
            # Normalize RGB to 0-1 range
            rgb_normalized = rgb.astype(np.float32) / 255.0
            r, g, b = cv2.split(rgb_normalized)
            
            # Calculate Intensity
            intensity = (r + g + b) / 3.0
            
            # Calculate Saturation
            minimum = np.minimum(np.minimum(r, g), b)
            saturation = 1 - (3 * minimum) / (r + g + b + 1e-10)
            
            # Calculate Hue
            numerator = 0.5 * ((r - g) + (r - b))
            denominator = np.sqrt((r - g)**2 + (r - b) * (g - b)) + 1e-10
            theta = np.arccos(np.clip(numerator / denominator, -1, 1))
            
            # Hue calculation
            hue = np.where(b <= g, theta, 2 * np.pi - theta)
            hue = hue / (2 * np.pi)  # Normalize to 0-1
            
            # Handle NaN values
            hue = np.nan_to_num(hue, nan=0.0)
            saturation = np.nan_to_num(saturation, nan=0.0)
            intensity = np.nan_to_num(intensity, nan=0.0)
            
            # Merge channels and convert to 8-bit
            hsi = cv2.merge((hue, saturation, intensity))
            return (hsi * 255).astype(np.uint8)
    
    async def convert_all_color_spaces(self, request, file: UploadFile):
        """Convert image to all supported color spaces"""
        try:
            # Save uploaded file
            file_path = os.path.join(self.uploads_path, file.filename)
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Load image
            img_bgr = cv2.imread(file_path)
            if img_bgr is None:
                raise ValueError("Invalid image file")
            
            # Convert to RGB for processing
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # Create comprehensive visualization
            fig = plt.figure(figsize=(20, 16))
            gs = gridspec.GridSpec(4, 5, hspace=0.3, wspace=0.3)
            
            # Original RGB
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.imshow(img_rgb)
            ax1.set_title('Original RGB', fontsize=12, fontweight='bold')
            ax1.axis('off')
            
            # 1. XYZ Color Space
            img_xyz = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2XYZ)
            ax2 = fig.add_subplot(gs[0, 1])
            # Normalize XYZ for display
            img_xyz_norm = cv2.normalize(img_xyz, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            ax2.imshow(img_xyz_norm)
            ax2.set_title('XYZ Color Space', fontsize=12, fontweight='bold')
            ax2.axis('off')
            
            # 2. LAB Color Space
            img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
            ax3 = fig.add_subplot(gs[0, 2])
            ax3.imshow(img_lab)
            ax3.set_title('CIE LAB Color Space', fontsize=12, fontweight='bold')
            ax3.axis('off')
            
            # 3. YCrCb Color Space
            img_ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
            ax4 = fig.add_subplot(gs[0, 3])
            ax4.imshow(img_ycrcb)
            ax4.set_title('YCrCb Color Space', fontsize=12, fontweight='bold')
            ax4.axis('off')
            
            # 4. YUV Color Space
            img_yuv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)
            ax5 = fig.add_subplot(gs[0, 4])
            ax5.imshow(img_yuv)
            ax5.set_title('YUV Color Space', fontsize=12, fontweight='bold')
            ax5.axis('off')
            
            # 5. YIQ Color Space (Manual Implementation)
            img_yiq = self.rgb_to_yiq_manual(img_rgb)
            ax6 = fig.add_subplot(gs[1, 0])
            ax6.imshow(img_yiq)
            ax6.set_title('YIQ Color Space (Manual)', fontsize=12, fontweight='bold')
            ax6.axis('off')
            
            # 6. HSI Color Space (Manual Implementation)
            img_hsi = self.rgb_to_hsi_manual(img_rgb)
            ax7 = fig.add_subplot(gs[1, 1])
            ax7.imshow(img_hsi)
            ax7.set_title('HSI Color Space (Manual)', fontsize=12, fontweight='bold')
            ax7.axis('off')
            
            # 7. HSV Color Space
            img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
            ax8 = fig.add_subplot(gs[1, 2])
            ax8.imshow(img_hsv)
            ax8.set_title('HSV Color Space', fontsize=12, fontweight='bold')
            ax8.axis('off')
            
            # 8. CIE Luv Color Space
            img_luv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Luv)
            ax9 = fig.add_subplot(gs[1, 3])
            # Normalize Luv for display
            img_luv_norm = cv2.normalize(img_luv, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            ax9.imshow(img_luv_norm)
            ax9.set_title('CIE Luv Color Space', fontsize=12, fontweight='bold')
            ax9.axis('off')
            
            # Component Analysis for RGB
            r, g, b = cv2.split(img_rgb)
            ax10 = fig.add_subplot(gs[2, 0])
            ax10.imshow(r, cmap='Reds')
            ax10.set_title('RGB - Red Component', fontsize=10)
            ax10.axis('off')
            
            ax11 = fig.add_subplot(gs[2, 1])
            ax11.imshow(g, cmap='Greens')
            ax11.set_title('RGB - Green Component', fontsize=10)
            ax11.axis('off')
            
            ax12 = fig.add_subplot(gs[2, 2])
            ax12.imshow(b, cmap='Blues')
            ax12.set_title('RGB - Blue Component', fontsize=10)
            ax12.axis('off')
            
            # Component Analysis for LAB
            l, a, b_lab = cv2.split(img_lab)
            ax13 = fig.add_subplot(gs[2, 3])
            ax13.imshow(l, cmap='gray')
            ax13.set_title('LAB - L Component', fontsize=10)
            ax13.axis('off')
            
            ax14 = fig.add_subplot(gs[2, 4])
            ax14.imshow(a, cmap='RdYlGn_r')
            ax14.set_title('LAB - A Component', fontsize=10)
            ax14.axis('off')
            
            # Component Analysis for HSV
            h, s, v = cv2.split(img_hsv)
            ax15 = fig.add_subplot(gs[3, 0])
            ax15.imshow(h, cmap='hsv')
            ax15.set_title('HSV - Hue Component', fontsize=10)
            ax15.axis('off')
            
            ax16 = fig.add_subplot(gs[3, 1])
            ax16.imshow(s, cmap='gray')
            ax16.set_title('HSV - Saturation Component', fontsize=10)
            ax16.axis('off')
            
            ax17 = fig.add_subplot(gs[3, 2])
            ax17.imshow(v, cmap='gray')
            ax17.set_title('HSV - Value Component', fontsize=10)
            ax17.axis('off')
            
            # YCrCb Components
            y, cr, cb = cv2.split(img_ycrcb)
            ax18 = fig.add_subplot(gs[3, 3])
            ax18.imshow(y, cmap='gray')
            ax18.set_title('YCrCb - Y Component', fontsize=10)
            ax18.axis('off')
            
            ax19 = fig.add_subplot(gs[3, 4])
            ax19.imshow(cr, cmap='Reds')
            ax19.set_title('YCrCb - Cr Component', fontsize=10)
            ax19.axis('off')
            
            plt.suptitle("Complete Color Space Analysis", fontsize=16, fontweight='bold')
            
            # Convert to base64
            plot_url = self.save_plot_as_base64(fig)
            
            # Prepare analysis results
            color_spaces = [
                {'name': 'RGB', 'description': 'Additive color model for displays', 'components': 'Red, Green, Blue', 'range': '0-255 each'},
                {'name': 'XYZ', 'description': 'CIE standard color space', 'components': 'X, Y, Z tristimulus', 'range': 'Device dependent'},
                {'name': 'LAB', 'description': 'Perceptually uniform color space', 'components': 'L (Lightness), A (Green-Red), B (Blue-Yellow)', 'range': 'L: 0-100, A,B: -128 to +127'},
                {'name': 'YCrCb', 'description': 'Used in JPEG compression', 'components': 'Y (Luminance), Cr (Red-diff), Cb (Blue-diff)', 'range': 'Y: 16-235, Cr,Cb: 16-240'},
                {'name': 'YUV', 'description': 'Used in PAL TV systems', 'components': 'Y (Luminance), U, V (Chrominance)', 'range': 'Y: 0-1, U,V: -0.5 to +0.5'},
                {'name': 'YIQ', 'description': 'Used in NTSC TV systems', 'components': 'Y (Luminance), I (In-phase), Q (Quadrature)', 'range': 'Y: 0-1, I: -0.6 to +0.6, Q: -0.5 to +0.5'},
                {'name': 'HSI', 'description': 'Intuitive for human perception', 'components': 'H (Hue), S (Saturation), I (Intensity)', 'range': 'H: 0-360°, S,I: 0-1'},
                {'name': 'HSV', 'description': 'Used in computer graphics', 'components': 'H (Hue), S (Saturation), V (Value)', 'range': 'H: 0-360°, S,V: 0-1'},
                {'name': 'CIE Luv', 'description': 'Alternative to LAB for additive colors', 'components': 'L (Lightness), u, v (Chromaticity)', 'range': 'L: 0-100, u,v: variable'}
            ]
            
            # Clean up
            os.remove(file_path)
            
            return self.templates.TemplateResponse("modules/modul7/result.html", {
                "request": request,
                "title": "Complete Color Space Analysis",
                "plot_url": plot_url,
                "analysis_type": "Color Space Conversion",
                "color_spaces": color_spaces,
                "success": True
            })
            
        except Exception as e:
            return self.templates.TemplateResponse("modules/modul7/result.html", {
                "request": request,
                "title": "Error",
                "error": str(e),
                "success": False
            })
    
    async def analyze_specific_color_space(self, request, file: UploadFile, color_space: str):
        """Analyze specific color space in detail"""
        try:
            # Save uploaded file
            file_path = os.path.join(self.uploads_path, file.filename)
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Load image
            img_bgr = cv2.imread(file_path)
            if img_bgr is None:
                raise ValueError("Invalid image file")
            
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # Create detailed analysis for specific color space
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Original image
            axes[0, 0].imshow(img_rgb)
            axes[0, 0].set_title('Original RGB Image', fontweight='bold')
            axes[0, 0].axis('off')
            
            if color_space.lower() == 'xyz':
                img_converted = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2XYZ)
                img_display = cv2.normalize(img_converted, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                components = cv2.split(img_converted)
                comp_names = ['X Component', 'Y Component', 'Z Component']
                colormaps = ['Reds', 'Greens', 'Blues']
                
            elif color_space.lower() == 'lab':
                img_converted = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
                img_display = img_converted
                components = cv2.split(img_converted)
                comp_names = ['L Component (Lightness)', 'A Component (Green-Red)', 'B Component (Blue-Yellow)']
                colormaps = ['gray', 'RdYlGn_r', 'coolwarm']
                
            elif color_space.lower() == 'ycrcb':
                img_converted = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
                img_display = img_converted
                components = cv2.split(img_converted)
                comp_names = ['Y Component (Luminance)', 'Cr Component (Red-diff)', 'Cb Component (Blue-diff)']
                colormaps = ['gray', 'Reds', 'Blues']
                
            elif color_space.lower() == 'yuv':
                img_converted = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)
                img_display = img_converted
                components = cv2.split(img_converted)
                comp_names = ['Y Component (Luminance)', 'U Component', 'V Component']
                colormaps = ['gray', 'Blues', 'Reds']
                
            elif color_space.lower() == 'yiq':
                img_converted = self.rgb_to_yiq_manual(img_rgb)
                img_display = img_converted
                components = cv2.split(img_converted)
                comp_names = ['Y Component (Luminance)', 'I Component (In-phase)', 'Q Component (Quadrature)']
                colormaps = ['gray', 'RdYlBu', 'RdYlGn']
                
            elif color_space.lower() == 'hsi':
                img_converted = self.rgb_to_hsi_manual(img_rgb)
                img_display = img_converted
                components = cv2.split(img_converted)
                comp_names = ['H Component (Hue)', 'S Component (Saturation)', 'I Component (Intensity)']
                colormaps = ['hsv', 'gray', 'gray']
                
            elif color_space.lower() == 'hsv':
                img_converted = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
                img_display = img_converted
                components = cv2.split(img_converted)
                comp_names = ['H Component (Hue)', 'S Component (Saturation)', 'V Component (Value)']
                colormaps = ['hsv', 'gray', 'gray']
                
            elif color_space.lower() == 'luv':
                img_converted = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Luv)
                img_display = cv2.normalize(img_converted, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                components = cv2.split(img_converted)
                comp_names = ['L Component (Lightness)', 'U Component', 'V Component']
                colormaps = ['gray', 'RdYlBu', 'RdYlGn']
            
            # Display converted image
            axes[0, 1].imshow(img_display)
            axes[0, 1].set_title(f'{color_space.upper()} Color Space', fontweight='bold')
            axes[0, 1].axis('off')
            
            # Display histograms
            axes[0, 2].hist(img_rgb.ravel(), bins=256, alpha=0.7, label='RGB Combined')
            axes[0, 2].set_title('RGB Histogram')
            axes[0, 2].set_xlabel('Pixel Value')
            axes[0, 2].set_ylabel('Frequency')
            
            # Display individual components
            for i in range(3):
                axes[1, i].imshow(components[i], cmap=colormaps[i])
                axes[1, i].set_title(comp_names[i])
                axes[1, i].axis('off')
            
            plt.suptitle(f'Detailed {color_space.upper()} Analysis', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Convert to base64
            plot_url = self.save_plot_as_base64(fig)
            
            # Clean up
            os.remove(file_path)
            
            return self.templates.TemplateResponse("modules/modul7/result.html", {
                "request": request,
                "title": f"{color_space.upper()} Color Space Analysis",
                "plot_url": plot_url,
                "analysis_type": f"{color_space.upper()} Detailed Analysis",
                "color_space": color_space.upper(),
                "success": True
            })
            
        except Exception as e:
            return self.templates.TemplateResponse("modules/modul7/result.html", {
                "request": request,
                "title": "Error",
                "error": str(e),
                "success": False
            })
    
    async def demo_color_conversion(self, request):
        """Demo color conversion dengan dataset bawaan"""
        try:
            demo_images = ["lena.png", "peppers.png"]
            demo_results = []
            
            for img_name in demo_images:
                img_path = os.path.join(self.dataset_path, img_name)
                if os.path.exists(img_path):
                    # Load image
                    img_bgr = cv2.imread(img_path)
                    if img_bgr is None:
                        continue
                    
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    
                    # Convert to various color spaces
                    conversions = {}
                    conversions['XYZ'] = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2XYZ)
                    conversions['LAB'] = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
                    conversions['YCrCb'] = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
                    conversions['YUV'] = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)
                    conversions['HSV'] = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
                    conversions['YIQ'] = self.rgb_to_yiq_manual(img_rgb)
                    conversions['HSI'] = self.rgb_to_hsi_manual(img_rgb)
                    conversions['Luv'] = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Luv)
                    
                    demo_results.append({
                        "image": img_name,
                        "total_conversions": len(conversions),
                        "size": f"{img_rgb.shape[1]}x{img_rgb.shape[0]}",
                        "channels": img_rgb.shape[2] if len(img_rgb.shape) > 2 else 1
                    })
            
            return self.templates.TemplateResponse("modules/modul7/demo.html", {
                "request": request,
                "title": "Color Space Demo Results",
                "demo_results": demo_results,
                "success": True
            })
            
        except Exception as e:
            return self.templates.TemplateResponse("modules/modul7/demo.html", {
                "request": request,
                "title": "Demo Error",
                "error": str(e),
                "success": False
            })
    
    async def demo_page(self, request):
        """Demo page untuk Modul 7"""
        return self.templates.TemplateResponse("modules/modul7/demo.html", {
            "request": request,
            "title": "Demo - Color Space Conversion"
        })
    
    async def demo_analysis(self, request, demo_image: str, analysis_type: str):
        """Process demo analysis dengan gambar dan analisis yang dipilih"""
        try:
            # Load demo image
            image_path = os.path.join(self.dataset_path, demo_image)
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Demo image {demo_image} tidak ditemukan")
            
            img_bgr = cv2.imread(image_path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # Generate analysis based on type
            if analysis_type == "all_spaces":
                return await self._demo_all_spaces(request, img_rgb, demo_image)
            elif analysis_type == "luminance_comparison":
                return await self._demo_luminance_comparison(request, img_rgb, demo_image)
            elif analysis_type == "chrominance_analysis":
                return await self._demo_chrominance_analysis(request, img_rgb, demo_image)
            elif analysis_type == "manual_implementations":
                return await self._demo_manual_implementations(request, img_rgb, demo_image)
            elif analysis_type == "perceptual_analysis":
                return await self._demo_perceptual_analysis(request, img_rgb, demo_image)
            else:
                raise ValueError(f"Unknown analysis type: {analysis_type}")
                
        except Exception as e:
            return self.templates.TemplateResponse("modules/modul7/result.html", {
                "request": request,
                "title": "Demo Analysis Error",
                "error": str(e),
                "success": False
            })
    
    async def _demo_all_spaces(self, request, img_rgb, image_name):
        """Demo untuk semua color spaces"""
        conversions = {}
        color_spaces = ['RGB', 'XYZ', 'LAB', 'YCrCb', 'YUV', 'YIQ', 'HSI', 'HSV', 'Luv']
        
        # Convert to all color spaces
        conversions['RGB'] = img_rgb
        conversions['XYZ'] = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2XYZ)
        conversions['LAB'] = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        conversions['YCrCb'] = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
        conversions['YUV'] = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)
        conversions['YIQ'] = self.rgb_to_yiq_manual(img_rgb)
        conversions['HSI'] = self.rgb_to_hsi_manual(img_rgb)
        conversions['HSV'] = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        conversions['Luv'] = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Luv)
        
        # Create visualization
        fig = plt.figure(figsize=(18, 15))
        gs = gridspec.GridSpec(3, 3, figure=fig)
        
        for i, space in enumerate(color_spaces):
            ax = fig.add_subplot(gs[i//3, i%3])
            img_display = self.normalize_for_display(conversions[space])
            ax.imshow(img_display)
            ax.set_title(f'{space} Color Space', fontsize=12, fontweight='bold')
            ax.axis('off')
        
        plt.tight_layout()
        plot_base64 = self.save_plot_as_base64(fig)
        plt.close(fig)
        
        return self.templates.TemplateResponse("modules/modul7/result.html", {
            "request": request,
            "title": f"Demo: All Color Spaces - {image_name}",
            "analysis_type": "Semua Model Warna (9 Model)",
            "original_image": image_name,
            "plot_base64": plot_base64,
            "color_spaces": color_spaces,
            "conversions": conversions,
            "success": True
        })
    
    async def _demo_luminance_comparison(self, request, img_rgb, image_name):
        """Demo perbandingan luminance components"""
        # Extract luminance from different color spaces
        xyz = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2XYZ)
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
        yuv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)
        yiq = self.rgb_to_yiq_manual(img_rgb)
        hsi = self.rgb_to_hsi_manual(img_rgb)
        
        # Create luminance comparison
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        luminance_components = [
            (xyz[:,:,1], 'XYZ - Y (Luminance)'),
            (lab[:,:,0], 'LAB - L (Lightness)'),
            (ycrcb[:,:,0], 'YCrCb - Y (Luma)'),
            (yuv[:,:,0], 'YUV - Y (Luma)'),
            (yiq[:,:,0], 'YIQ - Y (Luma)'),
            (hsi[:,:,2], 'HSI - I (Intensity)')
        ]
        
        for i, (component, title) in enumerate(luminance_components):
            ax = axes[i//3, i%3]
            ax.imshow(component, cmap='gray')
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.axis('off')
        
        plt.tight_layout()
        plot_base64 = self.save_plot_as_base64(fig)
        plt.close(fig)
        
        return self.templates.TemplateResponse("modules/modul7/result.html", {
            "request": request,
            "title": f"Demo: Luminance Comparison - {image_name}",
            "analysis_type": "Perbandingan Luminansi",
            "original_image": image_name,
            "plot_base64": plot_base64,
            "success": True
        })
    
    async def _demo_chrominance_analysis(self, request, img_rgb, image_name):
        """Demo analisis chrominance components"""
        # Extract chrominance from different color spaces
        ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
        yuv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)
        yiq = self.rgb_to_yiq_manual(img_rgb)
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        
        # Create chrominance visualization
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        chrominance_components = [
            (ycrcb[:,:,1], 'YCrCb - Cr'),
            (ycrcb[:,:,2], 'YCrCb - Cb'),
            (yuv[:,:,1], 'YUV - U'),
            (yuv[:,:,2], 'YUV - V'),
            (yiq[:,:,1], 'YIQ - I'),
            (yiq[:,:,2], 'YIQ - Q'),
            (lab[:,:,1], 'LAB - a*'),
            (lab[:,:,2], 'LAB - b*')
        ]
        
        for i, (component, title) in enumerate(chrominance_components):
            ax = axes[i//4, i%4]
            ax.imshow(component, cmap='RdBu_r')
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.axis('off')
        
        plt.tight_layout()
        plot_base64 = self.save_plot_as_base64(fig)
        plt.close(fig)
        
        return self.templates.TemplateResponse("modules/modul7/result.html", {
            "request": request,
            "title": f"Demo: Chrominance Analysis - {image_name}",
            "analysis_type": "Analisis Chrominance",
            "original_image": image_name,
            "plot_base64": plot_base64,
            "success": True
        })
    
    async def _demo_manual_implementations(self, request, img_rgb, image_name):
        """Demo implementasi manual YIQ dan HSI"""
        # Manual implementations
        yiq_manual = self.rgb_to_yiq_manual(img_rgb)
        hsi_manual = self.rgb_to_hsi_manual(img_rgb)
        
        # OpenCV implementations for comparison
        hsv_opencv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        
        # Create comparison
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # YIQ Manual
        axes[0,0].imshow(yiq_manual[:,:,0], cmap='gray')
        axes[0,0].set_title('YIQ-Y (Manual)', fontweight='bold')
        axes[0,0].axis('off')
        
        axes[0,1].imshow(yiq_manual[:,:,1], cmap='RdBu_r')
        axes[0,1].set_title('YIQ-I (Manual)', fontweight='bold')
        axes[0,1].axis('off')
        
        axes[0,2].imshow(yiq_manual[:,:,2], cmap='RdBu_r')
        axes[0,2].set_title('YIQ-Q (Manual)', fontweight='bold')
        axes[0,2].axis('off')
        
        axes[0,3].imshow(self.normalize_for_display(yiq_manual))
        axes[0,3].set_title('YIQ Combined', fontweight='bold')
        axes[0,3].axis('off')
        
        # HSI Manual
        axes[1,0].imshow(hsi_manual[:,:,0], cmap='hsv')
        axes[1,0].set_title('HSI-H (Manual)', fontweight='bold')
        axes[1,0].axis('off')
        
        axes[1,1].imshow(hsi_manual[:,:,1], cmap='gray')
        axes[1,1].set_title('HSI-S (Manual)', fontweight='bold')
        axes[1,1].axis('off')
        
        axes[1,2].imshow(hsi_manual[:,:,2], cmap='gray')
        axes[1,2].set_title('HSI-I (Manual)', fontweight='bold')
        axes[1,2].axis('off')
        
        axes[1,3].imshow(self.normalize_for_display(hsi_manual))
        axes[1,3].set_title('HSI Combined', fontweight='bold')
        axes[1,3].axis('off')
        
        plt.tight_layout()
        plot_base64 = self.save_plot_as_base64(fig)
        plt.close(fig)
        
        return self.templates.TemplateResponse("modules/modul7/result.html", {
            "request": request,
            "title": f"Demo: Manual Implementations - {image_name}",
            "analysis_type": "Implementasi Manual (YIQ & HSI)",
            "original_image": image_name,
            "plot_base64": plot_base64,
            "success": True
        })
    
    async def _demo_perceptual_analysis(self, request, img_rgb, image_name):
        """Demo analisis perceptual LAB vs Luv"""
        # Convert to perceptual color spaces
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        luv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Luv)
        
        # Create perceptual comparison
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # LAB components
        axes[0,0].imshow(lab[:,:,0], cmap='gray')
        axes[0,0].set_title('LAB - L* (Lightness)', fontweight='bold')
        axes[0,0].axis('off')
        
        axes[0,1].imshow(lab[:,:,1], cmap='RdGy_r')
        axes[0,1].set_title('LAB - a* (Green-Red)', fontweight='bold')
        axes[0,1].axis('off')
        
        axes[0,2].imshow(lab[:,:,2], cmap='YlBl_r')
        axes[0,2].set_title('LAB - b* (Blue-Yellow)', fontweight='bold')
        axes[0,2].axis('off')
        
        axes[0,3].imshow(self.normalize_for_display(lab))
        axes[0,3].set_title('LAB Combined', fontweight='bold')
        axes[0,3].axis('off')
        
        # Luv components
        axes[1,0].imshow(luv[:,:,0], cmap='gray')
        axes[1,0].set_title('Luv - L* (Lightness)', fontweight='bold')
        axes[1,0].axis('off')
        
        axes[1,1].imshow(luv[:,:,1], cmap='RdBu_r')
        axes[1,1].set_title('Luv - u*', fontweight='bold')
        axes[1,1].axis('off')
        
        axes[1,2].imshow(luv[:,:,2], cmap='RdBu_r')
        axes[1,2].set_title('Luv - v*', fontweight='bold')
        axes[1,2].axis('off')
        
        axes[1,3].imshow(self.normalize_for_display(luv))
        axes[1,3].set_title('Luv Combined', fontweight='bold')
        axes[1,3].axis('off')
        
        plt.tight_layout()
        plot_base64 = self.save_plot_as_base64(fig)
        plt.close(fig)
        
        return self.templates.TemplateResponse("modules/modul7/result.html", {
            "request": request,
            "title": f"Demo: Perceptual Analysis - {image_name}",
            "analysis_type": "Analisis Perceptual (Lab vs Luv)",
            "original_image": image_name,
            "plot_base64": plot_base64,
            "success": True
        })