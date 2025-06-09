"""
Modul 5: Pengolahan Citra Digital - Advanced Analysis
Menggabungkan Freeman Chain Code, Canny Edge Detection, dan Integral Projection
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

class Modul5:
    """
    Class untuk Modul 5 - Advanced Analysis
    Compatible dengan FastAPI framework
    """
    
    def __init__(self, templates: Jinja2Templates):
        self.templates = templates
        self.dataset_path = "static/dataset/"
        self.uploads_path = "static/uploads/"
        self.results_path = "static/results/"
        
        # Ensure results directory exists
        os.makedirs(self.results_path, exist_ok=True)
    
    async def home(self, request):
        """Home page untuk Modul 5"""
        return self.templates.TemplateResponse("modules/modul5/home.html", {
            "request": request,
            "title": "Modul 5: Advanced Analysis"
        })
    
    def generate_freeman_chain_code(self, contour):
        """
        Menghasilkan Kode Rantai Freeman 8-arah dari kontur OpenCV.
        """
        chain_code = []
        if len(contour) < 2:
            return chain_code

        directions = {
            (1, 0): 0, (1, 1): 1, (0, 1): 2, (-1, 1): 3,
            (-1, 0): 4, (-1, -1): 5, (0, -1): 6, (1, -1): 7
        }

        for i in range(len(contour)):
            p1 = contour[i][0]
            p2 = contour[(i + 1) % len(contour)][0]

            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]

            norm_dx = np.sign(dx)
            norm_dy = np.sign(dy)

            code = directions.get((norm_dx, norm_dy))
            if code is not None:
                chain_code.append(code)

        return chain_code
    
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
    
    async def process_freeman_chain_code(self, request, file: UploadFile):
        """Process Freeman Chain Code analysis"""
        try:
            # Save uploaded file
            file_path = os.path.join(self.uploads_path, file.filename)
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Load and process image
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError("Invalid image file")
            
            # Binarization
            _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours
            contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            # Process results
            result = {
                'original_image': img,
                'binary_image': binary_img,
                'contours': contours,
                'chain_code': [],
                'contour_image': None
            }
            
            if contours:
                # Get largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Draw contour
                img_contour_display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                cv2.drawContours(img_contour_display, [largest_contour], -1, (0, 255, 0), 2)
                result['contour_image'] = img_contour_display
                
                # Generate chain code
                chain_code_result = self.generate_freeman_chain_code(largest_contour)
                result['chain_code'] = chain_code_result
            
            # Create visualization
            fig, axs = plt.subplots(2, 2, figsize=(12, 10))
            
            # Original image
            axs[0, 0].imshow(img, cmap='gray')
            axs[0, 0].set_title('Original Image')
            axs[0, 0].axis('off')
            
            # Binary image
            axs[0, 1].imshow(binary_img, cmap='gray')
            axs[0, 1].set_title('Binary Image')
            axs[0, 1].axis('off')
            
            # Contour image
            if result['contour_image'] is not None:
                img_rgb = cv2.cvtColor(result['contour_image'], cv2.COLOR_BGR2RGB)
                axs[1, 0].imshow(img_rgb)
                axs[1, 0].set_title('Detected Contour')
            else:
                axs[1, 0].text(0.5, 0.5, 'No contour found', ha='center', va='center')
                axs[1, 0].set_title('Contour Detection')
            axs[1, 0].axis('off')
            
            # Chain code text
            axs[1, 1].axis('off')
            if result['chain_code']:
                chain_code_str = f"Chain Code Length: {len(result['chain_code'])}\n\n"
                # Format chain code with line breaks
                codes_per_line = 20
                for i in range(0, len(result['chain_code']), codes_per_line):
                    line_codes = result['chain_code'][i:i+codes_per_line]
                    chain_code_str += " ".join(map(str, line_codes)) + "\n"
            else:
                chain_code_str = "No contour found.\nUnable to generate chain code."
            
            axs[1, 1].text(0.05, 0.95, chain_code_str, ha='left', va='top', 
                          fontsize=10, transform=axs[1, 1].transAxes, fontfamily='monospace')
            axs[1, 1].set_title('Freeman Chain Code')
            
            plt.tight_layout()
            plt.suptitle("Freeman Chain Code Analysis", fontsize=16, y=0.98)
            
            # Convert to base64
            plot_url = self.save_plot_as_base64(fig)
            
            # Clean up
            os.remove(file_path)
            
            return self.templates.TemplateResponse("modules/modul5/result.html", {
                "request": request,
                "title": "Freeman Chain Code Result",
                "plot_url": plot_url,
                "analysis_type": "Freeman Chain Code",
                "chain_code_length": len(result['chain_code']) if result['chain_code'] else 0,
                "contours_found": len(contours),
                "success": True
            })
            
        except Exception as e:
            return self.templates.TemplateResponse("modules/modul5/result.html", {
                "request": request,
                "title": "Error",
                "error": str(e),
                "success": False
            })
    
    async def process_canny_edge_detection(self, request, file: UploadFile, low_threshold: int, high_threshold: int):
        """Process Canny Edge Detection"""
        try:
            # Save uploaded file
            file_path = os.path.join(self.uploads_path, file.filename)
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Load and process image
            img = cv2.imread(file_path)
            if img is None:
                raise ValueError("Invalid image file")
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Canny edge detection
            edges = cv2.Canny(blurred, low_threshold, high_threshold)
            
            # Create visualization
            fig = plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title('Original Image')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(blurred, cmap='gray')
            plt.title('Grayscale + Gaussian Blur')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(edges, cmap='gray')
            plt.title(f'Canny Edges (Th={low_threshold},{high_threshold})')
            plt.axis('off')
            
            plt.tight_layout()
            plt.suptitle("Canny Edge Detection Analysis", fontsize=16, y=1.02)
            
            # Convert to base64
            plot_url = self.save_plot_as_base64(fig)
            
            # Clean up
            os.remove(file_path)
            
            return self.templates.TemplateResponse("modules/modul5/result.html", {
                "request": request,
                "title": "Canny Edge Detection Result",
                "plot_url": plot_url,
                "analysis_type": "Canny Edge Detection",
                "low_threshold": low_threshold,
                "high_threshold": high_threshold,
                "success": True
            })
            
        except Exception as e:
            return self.templates.TemplateResponse("modules/modul5/result.html", {
                "request": request,
                "title": "Error",
                "error": str(e),
                "success": False
            })
    
    async def process_integral_projection(self, request, file: UploadFile):
        """Process Integral Projection"""
        try:
            # Save uploaded file
            file_path = os.path.join(self.uploads_path, file.filename)
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Load and process image
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError("Invalid image file")
            
            # Binarization
            _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Normalize to 0-1
            binary_norm = binary_img / 255.0
            
            # Calculate projections
            horizontal_projection = np.sum(binary_norm, axis=0)
            vertical_projection = np.sum(binary_norm, axis=1)
            
            height, width = binary_norm.shape
            
            # Create visualization
            fig = plt.figure(figsize=(12, 8))
            gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                                  left=0.1, right=0.9, bottom=0.1, top=0.9,
                                  wspace=0.05, hspace=0.05)
            
            # Main image
            ax_img = fig.add_subplot(gs[1, 0])
            ax_img.imshow(binary_norm, cmap='gray', aspect='auto')
            ax_img.set_title('Binary Image')
            ax_img.set_xlabel('Column Index')
            ax_img.set_ylabel('Row Index')
            
            # Horizontal projection
            ax_hproj = fig.add_subplot(gs[0, 0], sharex=ax_img)
            ax_hproj.plot(np.arange(width), horizontal_projection, color='blue')
            ax_hproj.set_title('Horizontal Projection')
            ax_hproj.set_ylabel('Pixel Count')
            plt.setp(ax_hproj.get_xticklabels(), visible=False)
            ax_hproj.grid(axis='y', linestyle='--', alpha=0.6)
            
            # Vertical projection
            ax_vproj = fig.add_subplot(gs[1, 1], sharey=ax_img)
            ax_vproj.plot(vertical_projection, np.arange(height), color='red')
            ax_vproj.set_title('Vertical Projection')
            ax_vproj.set_xlabel('Pixel Count')
            ax_vproj.invert_yaxis()
            plt.setp(ax_vproj.get_yticklabels(), visible=False)
            ax_vproj.grid(axis='x', linestyle='--', alpha=0.6)
            
            plt.suptitle("Integral Projection Analysis", fontsize=14)
            
            # Convert to base64
            plot_url = self.save_plot_as_base64(fig)
            
            # Clean up
            os.remove(file_path)
            
            return self.templates.TemplateResponse("modules/modul5/result.html", {
                "request": request,
                "title": "Integral Projection Result",
                "plot_url": plot_url,
                "analysis_type": "Integral Projection",
                "image_size": f"{width} x {height}",
                "success": True
            })
            
        except Exception as e:
            return self.templates.TemplateResponse("modules/modul5/result.html", {
                "request": request,
                "title": "Error",
                "error": str(e),
                "success": False
            })
    
    async def demo_analysis(self, request):
        """Demo analysis menggunakan dataset bawaan"""
        try:
            demo_images = ["hurufA.png", "cameraman.png", "tier1.png"]
            results = []
            
            for img_name in demo_images:
                img_path = os.path.join(self.dataset_path, img_name)
                if os.path.exists(img_path):
                    # Process each demo image with different technique
                    if "hurufA" in img_name:
                        # Freeman Chain Code demo
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
                        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                        
                        chain_length = 0
                        if contours:
                            largest_contour = max(contours, key=cv2.contourArea)
                            chain_code = self.generate_freeman_chain_code(largest_contour)
                            chain_length = len(chain_code)
                        
                        results.append({
                            "image": img_name,
                            "technique": "Freeman Chain Code",
                            "result": f"Chain length: {chain_length}, Contours: {len(contours)}"
                        })
                    
                    elif "cameraman" in img_name:
                        # Canny Edge Detection demo
                        img = cv2.imread(img_path)
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                        edges = cv2.Canny(blurred, 50, 150)
                        edge_pixels = np.sum(edges > 0)
                        
                        results.append({
                            "image": img_name,
                            "technique": "Canny Edge Detection",
                            "result": f"Edge pixels: {edge_pixels}, Threshold: 50-150"
                        })
                    
                    elif "tier1" in img_name:
                        # Integral Projection demo
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                        binary_norm = binary_img / 255.0
                        h_proj = np.sum(binary_norm, axis=0)
                        v_proj = np.sum(binary_norm, axis=1)
                        
                        results.append({
                            "image": img_name,
                            "technique": "Integral Projection",
                            "result": f"H-proj max: {np.max(h_proj):.1f}, V-proj max: {np.max(v_proj):.1f}"
                        })
            
            return self.templates.TemplateResponse("modules/modul5/demo.html", {
                "request": request,
                "title": "Demo Analysis Results",
                "results": results,
                "success": True
            })
            
        except Exception as e:
            return self.templates.TemplateResponse("modules/modul5/demo.html", {
                "request": request,
                "title": "Demo Error",
                "error": str(e),
                "success": False
            })
    
    async def complete_analysis(self, request, file: UploadFile):
        """Complete analysis combining all techniques"""
        try:
            # Save uploaded file
            file_path = os.path.join(self.uploads_path, file.filename)
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Load image
            img_color = cv2.imread(file_path)
            img_gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            
            if img_color is None or img_gray is None:
                raise ValueError("Invalid image file")
            
            # Create comprehensive analysis plot
            fig = plt.figure(figsize=(16, 12))
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
            
            # 1. Original image
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
            ax1.set_title('Original Image')
            ax1.axis('off')
            
            # 2. Freeman Chain Code
            _, binary_img = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.imshow(binary_img, cmap='gray')
            ax2.set_title(f'Binary Image ({len(contours)} contours)')
            ax2.axis('off')
            
            # Draw contours if found
            if contours:
                img_contour = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
                cv2.drawContours(img_contour, contours, -1, (0, 255, 0), 2)
                ax3 = fig.add_subplot(gs[0, 2])
                ax3.imshow(cv2.cvtColor(img_contour, cv2.COLOR_BGR2RGB))
                
                # Calculate chain code for largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                chain_code = self.generate_freeman_chain_code(largest_contour)
                ax3.set_title(f'Contours (Chain length: {len(chain_code)})')
                ax3.axis('off')
            
            # 3. Canny Edge Detection
            gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            ax4 = fig.add_subplot(gs[1, 0])
            ax4.imshow(blurred, cmap='gray')
            ax4.set_title('Gaussian Blur')
            ax4.axis('off')
            
            ax5 = fig.add_subplot(gs[1, 1])
            ax5.imshow(edges, cmap='gray')
            ax5.set_title('Canny Edges')
            ax5.axis('off')
            
            # 4. Integral Projection
            _, binary_proj = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            binary_norm = binary_proj / 255.0
            h_proj = np.sum(binary_norm, axis=0)
            v_proj = np.sum(binary_norm, axis=1)
            
            ax6 = fig.add_subplot(gs[1, 2])
            ax6.imshow(binary_norm, cmap='gray')
            ax6.set_title('Binary for Projection')
            ax6.axis('off')
            
            # Projection plots
            ax7 = fig.add_subplot(gs[2, 0])
            ax7.plot(h_proj)
            ax7.set_title('Horizontal Projection')
            ax7.set_xlabel('Column')
            ax7.set_ylabel('Pixel Count')
            
            ax8 = fig.add_subplot(gs[2, 1])
            ax8.plot(v_proj, range(len(v_proj)))
            ax8.set_title('Vertical Projection')
            ax8.set_xlabel('Pixel Count')
            ax8.set_ylabel('Row')
            ax8.invert_yaxis()
            
            # Summary text
            ax9 = fig.add_subplot(gs[2, 2])
            ax9.axis('off')
            summary_text = f"""Complete Analysis Summary:

Freeman Chain Code:
- Contours found: {len(contours)}
- Chain length: {len(chain_code) if contours else 0}

Canny Edge Detection:
- Edge pixels: {np.sum(edges > 0)}
- Thresholds: 50-150

Integral Projection:
- H-proj max: {np.max(h_proj):.1f}
- V-proj max: {np.max(v_proj):.1f}
- Image size: {img_gray.shape[1]}x{img_gray.shape[0]}"""
            
            ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
            ax9.set_title('Analysis Summary')
            
            plt.suptitle("Complete Advanced Analysis", fontsize=16)
            
            # Convert to base64
            plot_url = self.save_plot_as_base64(fig)
            
            # Clean up
            os.remove(file_path)
            
            return self.templates.TemplateResponse("modules/modul5/result.html", {
                "request": request,
                "title": "Complete Analysis Result",
                "plot_url": plot_url,
                "analysis_type": "Complete Analysis",
                "contours_found": len(contours),
                "chain_code_length": len(chain_code) if contours else 0,
                "edge_pixels": int(np.sum(edges > 0)),
                "success": True
            })
            
        except Exception as e:
            return self.templates.TemplateResponse("modules/modul5/result.html", {
                "request": request,
                "title": "Error",
                "error": str(e),
                "success": False
            })