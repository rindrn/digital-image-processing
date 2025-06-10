"""
Modul 6: Pengolahan Citra Digital - Image Compression Analysis
Analisis kompresi JPEG dan PNG dengan evaluasi kualitas
Compatible dengan FastAPI Framework
Author: RINDI INDRIANI - 231511030
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
from fastapi import UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from skimage.metrics import structural_similarity as ssim
import os
import io
import base64
from PIL import Image
import subprocess

class Modul6:
    """
    Class untuk Modul 6 - Image Compression Analysis
    Compatible dengan FastAPI framework
    """
    
    def __init__(self, templates: Jinja2Templates):
        self.templates = templates
        self.dataset_path = "static/dataset/"
        self.uploads_path = "static/uploads/"
        self.results_path = "static/compression_results/"
        
        # Ensure results directory exists
        os.makedirs(self.results_path, exist_ok=True)
    
    async def home(self, request):
        """Home page untuk Modul 6"""
        return self.templates.TemplateResponse("modules/modul6/home.html", {
            "request": request,
            "title": "Modul 6: Image Compression Analysis"
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
    
    def calculate_image_metrics(self, img_original, img_compressed):
        """Calculate PSNR and SSIM metrics"""
        # Pastikan dimensi sama
        if img_original.shape != img_compressed.shape:
            return None, None
        
        # Hitung PSNR
        psnr_value = cv2.PSNR(img_original, img_compressed)
        
        # Tentukan win_size untuk SSIM
        min_dim = min(img_original.shape[:2])
        win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
        if win_size < 3:
            win_size = 3
        
        # Hitung SSIM
        try:
            if len(img_original.shape) == 3:  # Color image
                ssim_value = ssim(
                    img_original,
                    img_compressed,
                    channel_axis=2,
                    win_size=win_size,
                    data_range=img_original.max() - img_original.min()
                )
            else:  # Grayscale image
                ssim_value = ssim(
                    img_original,
                    img_compressed,
                    win_size=win_size,
                    data_range=img_original.max() - img_original.min()
                )
        except ValueError as e:
            print(f"Error calculating SSIM: {e}")
            ssim_value = None
        
        return psnr_value, ssim_value
    
    async def process_jpeg_compression(self, request, file: UploadFile):
        """Process JPEG compression analysis"""
        try:
            # Save uploaded file
            file_path = os.path.join(self.uploads_path, file.filename)
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Load image
            img_original_bgr = cv2.imread(file_path)
            if img_original_bgr is None:
                raise ValueError("Invalid image file")
            
            # Determine if color or grayscale
            if len(img_original_bgr.shape) == 3:
                is_color = True
                img_original_cv = cv2.cvtColor(img_original_bgr, cv2.COLOR_BGR2RGB)
                cmap_val = None
            else:
                is_color = False
                img_original_cv = img_original_bgr
                cmap_val = 'gray'
            
            original_size_bytes = os.path.getsize(file_path)
            jpeg_qualities = [95, 75, 50, 25, 10]
            results = []
            
            # Process different JPEG qualities
            for quality in jpeg_qualities:
                jpeg_path = os.path.join(self.results_path, f'compressed_jpeg_q{quality}.jpg')
                
                # Save with specific JPEG quality
                if is_color:
                    img_to_save = cv2.cvtColor(img_original_cv, cv2.COLOR_RGB2BGR)
                else:
                    img_to_save = img_original_cv
                
                cv2.imwrite(jpeg_path, img_to_save, [cv2.IMWRITE_JPEG_QUALITY, quality])
                compressed_size_bytes = os.path.getsize(jpeg_path)
                
                # Load compressed image
                img_compressed_bgr = cv2.imread(jpeg_path)
                if is_color:
                    img_compressed_cv = cv2.cvtColor(img_compressed_bgr, cv2.COLOR_BGR2RGB)
                else:
                    img_compressed_cv = cv2.imread(jpeg_path, cv2.IMREAD_GRAYSCALE)
                
                # Calculate metrics
                psnr_value, ssim_value = self.calculate_image_metrics(img_original_cv, img_compressed_cv)
                
                results.append({
                    'Quality': quality,
                    'FileSize (KB)': compressed_size_bytes / 1024,
                    'CompressionRatio': original_size_bytes / compressed_size_bytes if compressed_size_bytes > 0 else float('inf'),
                    'PSNR (dB)': psnr_value,
                    'SSIM': ssim_value
                })
            
            # Create visualization
            fig = plt.figure(figsize=(18, 10))
            
            # Display images comparison
            ax1 = plt.subplot(2, 3, 1)
            plt.imshow(img_original_cv, cmap=cmap_val)
            plt.title(f'Original ({original_size_bytes / 1024:.2f} KB)')
            plt.axis('off')
            
            # Show different quality levels
            quality_samples = [95, 50, 10]
            for i, quality in enumerate(quality_samples, 2):
                jpeg_path = os.path.join(self.results_path, f'compressed_jpeg_q{quality}.jpg')
                img_sample = cv2.imread(jpeg_path)
                if is_color:
                    img_sample = cv2.cvtColor(img_sample, cv2.COLOR_BGR2RGB)
                else:
                    img_sample = cv2.imread(jpeg_path, cv2.IMREAD_GRAYSCALE)
                
                plt.subplot(2, 3, i)
                plt.imshow(img_sample, cmap=cmap_val)
                
                # Find result for this quality
                result = next(r for r in results if r['Quality'] == quality)
                plt.title(f'JPEG Q{quality} ({result["FileSize (KB)"]:.2f} KB)\nPSNR: {result["PSNR (dB)"]:.2f} dB')
                plt.axis('off')
            
            # Plot compression metrics
            qualities = [r['Quality'] for r in results]
            file_sizes = [r['FileSize (KB)'] for r in results]
            psnr_values = [r['PSNR (dB)'] for r in results]
            
            plt.subplot(2, 3, 5)
            plt.plot(qualities, file_sizes, 'bo-', linewidth=2, markersize=8)
            plt.xlabel('JPEG Quality')
            plt.ylabel('File Size (KB)')
            plt.title('File Size vs Quality')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 3, 6)
            plt.plot(qualities, psnr_values, 'ro-', linewidth=2, markersize=8)
            plt.xlabel('JPEG Quality')
            plt.ylabel('PSNR (dB)')
            plt.title('PSNR vs Quality')
            plt.grid(True, alpha=0.3)
            
            plt.suptitle("JPEG Compression Analysis", fontsize=16)
            plt.tight_layout()
            
            # Convert to base64
            plot_url = self.save_plot_as_base64(fig)
            
            # Clean up
            os.remove(file_path)
            for quality in jpeg_qualities:
                jpeg_path = os.path.join(self.results_path, f'compressed_jpeg_q{quality}.jpg')
                if os.path.exists(jpeg_path):
                    os.remove(jpeg_path)
            
            return self.templates.TemplateResponse("modules/modul6/result.html", {
                "request": request,
                "title": "JPEG Compression Analysis",
                "plot_url": plot_url,
                "analysis_type": "JPEG Compression",
                "results": results,
                "original_size": round(original_size_bytes / 1024, 2),
                "success": True
            })
            
        except Exception as e:
            return self.templates.TemplateResponse("modules/modul6/result.html", {
                "request": request,
                "title": "Error",
                "error": str(e),
                "success": False
            })
    
    async def process_png_compression(self, request, file: UploadFile):
        """Process PNG compression analysis"""
        try:
            # Save uploaded file
            file_path = os.path.join(self.uploads_path, file.filename)
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Load image
            img_original_bgr = cv2.imread(file_path)
            if img_original_bgr is None:
                raise ValueError("Invalid image file")
            
            # Determine if color or grayscale
            if len(img_original_bgr.shape) == 3:
                is_color = True
                img_original_cv = cv2.cvtColor(img_original_bgr, cv2.COLOR_BGR2RGB)
                cmap_val = None
            else:
                is_color = False
                img_original_cv = img_original_bgr
                cmap_val = 'gray'
            
            original_size_bytes = os.path.getsize(file_path)
            png_compression_levels = [0, 1, 3, 6, 9]
            results = []
            
            # Process different PNG compression levels
            for level in png_compression_levels:
                png_path = os.path.join(self.results_path, f'compressed_png_level{level}.png')
                
                # Save with specific PNG compression level
                if is_color:
                    img_to_save = cv2.cvtColor(img_original_cv, cv2.COLOR_RGB2BGR)
                else:
                    img_to_save = img_original_cv
                
                cv2.imwrite(png_path, img_to_save, [cv2.IMWRITE_PNG_COMPRESSION, level])
                compressed_size_bytes = os.path.getsize(png_path)
                
                # Load compressed image
                img_compressed_bgr = cv2.imread(png_path)
                if is_color:
                    img_compressed_cv = cv2.cvtColor(img_compressed_bgr, cv2.COLOR_BGR2RGB)
                else:
                    img_compressed_cv = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
                
                # Verify lossless compression
                is_identical = np.array_equal(img_original_cv, img_compressed_cv)
                psnr_value, ssim_value = self.calculate_image_metrics(img_original_cv, img_compressed_cv)
                
                results.append({
                    'Level': level,
                    'FileSize (KB)': compressed_size_bytes / 1024,
                    'CompressionRatio': original_size_bytes / compressed_size_bytes if compressed_size_bytes > 0 else float('inf'),
                    'PSNR (dB)': 'Infinity (Lossless)' if is_identical else f'{psnr_value:.2f}',
                    'SSIM': ssim_value,
                    'Identical': is_identical
                })
            
            # Create visualization
            fig = plt.figure(figsize=(16, 8))
            
            # Display images comparison
            plt.subplot(2, 3, 1)
            plt.imshow(img_original_cv, cmap=cmap_val)
            plt.title(f'Original ({original_size_bytes / 1024:.2f} KB)')
            plt.axis('off')
            
            # Show different compression levels
            level_samples = [0, 6, 9]
            for i, level in enumerate(level_samples, 2):
                png_path = os.path.join(self.results_path, f'compressed_png_level{level}.png')
                img_sample = cv2.imread(png_path)
                if is_color:
                    img_sample = cv2.cvtColor(img_sample, cv2.COLOR_BGR2RGB)
                else:
                    img_sample = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
                
                plt.subplot(2, 3, i)
                plt.imshow(img_sample, cmap=cmap_val)
                
                # Find result for this level
                result = next(r for r in results if r['Level'] == level)
                plt.title(f'PNG Level {level} ({result["FileSize (KB)"]:.2f} KB)\nLossless Compression')
                plt.axis('off')
            
            # Plot compression metrics
            levels = [r['Level'] for r in results]
            file_sizes = [r['FileSize (KB)'] for r in results]
            compression_ratios = [r['CompressionRatio'] for r in results]
            
            plt.subplot(2, 3, 5)
            plt.plot(levels, file_sizes, 'go-', linewidth=2, markersize=8)
            plt.xlabel('PNG Compression Level')
            plt.ylabel('File Size (KB)')
            plt.title('File Size vs Compression Level')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 3, 6)
            plt.plot(levels, compression_ratios, 'mo-', linewidth=2, markersize=8)
            plt.xlabel('PNG Compression Level')
            plt.ylabel('Compression Ratio')
            plt.title('Compression Ratio vs Level')
            plt.grid(True, alpha=0.3)
            
            plt.suptitle("PNG Compression Analysis", fontsize=16)
            plt.tight_layout()
            
            # Convert to base64
            plot_url = self.save_plot_as_base64(fig)
            
            # Clean up
            os.remove(file_path)
            for level in png_compression_levels:
                png_path = os.path.join(self.results_path, f'compressed_png_level{level}.png')
                if os.path.exists(png_path):
                    os.remove(png_path)
            
            return self.templates.TemplateResponse("modules/modul6/result.html", {
                "request": request,
                "title": "PNG Compression Analysis",
                "plot_url": plot_url,
                "analysis_type": "PNG Compression",
                "results": results,
                "original_size": round(original_size_bytes / 1024, 2),
                "success": True
            })
            
        except Exception as e:
            return self.templates.TemplateResponse("modules/modul6/result.html", {
                "request": request,
                "title": "Error",
                "error": str(e),
                "success": False
            })
    
    async def compare_compression_methods(self, request, file: UploadFile):
        """Compare JPEG vs PNG compression methods"""
        try:
            # Save uploaded file
            file_path = os.path.join(self.uploads_path, file.filename)
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Load image
            img_original_bgr = cv2.imread(file_path)
            if img_original_bgr is None:
                raise ValueError("Invalid image file")
            
            # Determine if color or grayscale
            if len(img_original_bgr.shape) == 3:
                is_color = True
                img_original_cv = cv2.cvtColor(img_original_bgr, cv2.COLOR_BGR2RGB)
                cmap_val = None
            else:
                is_color = False
                img_original_cv = img_original_bgr
                cmap_val = 'gray'
            
            original_size_bytes = os.path.getsize(file_path)
            
            # Test specific quality/level combinations
            test_configs = [
                {'type': 'JPEG', 'quality': 95, 'color': 'red'},
                {'type': 'JPEG', 'quality': 75, 'color': 'orange'},
                {'type': 'JPEG', 'quality': 50, 'color': 'blue'},
                {'type': 'PNG', 'level': 1, 'color': 'green'},
                {'type': 'PNG', 'level': 6, 'color': 'purple'},
                {'type': 'PNG', 'level': 9, 'color': 'brown'}
            ]
            
            results = []
            
            for config in test_configs:
                if config['type'] == 'JPEG':
                    file_path_comp = os.path.join(self.results_path, f'compare_jpeg_q{config["quality"]}.jpg')
                    if is_color:
                        img_to_save = cv2.cvtColor(img_original_cv, cv2.COLOR_RGB2BGR)
                    else:
                        img_to_save = img_original_cv
                    cv2.imwrite(file_path_comp, img_to_save, [cv2.IMWRITE_JPEG_QUALITY, config['quality']])
                    label = f'JPEG Q{config["quality"]}'
                else:  # PNG
                    file_path_comp = os.path.join(self.results_path, f'compare_png_l{config["level"]}.png')
                    if is_color:
                        img_to_save = cv2.cvtColor(img_original_cv, cv2.COLOR_RGB2BGR)
                    else:
                        img_to_save = img_original_cv
                    cv2.imwrite(file_path_comp, img_to_save, [cv2.IMWRITE_PNG_COMPRESSION, config['level']])
                    label = f'PNG L{config["level"]}'
                
                compressed_size_bytes = os.path.getsize(file_path_comp)
                
                # Load compressed image
                img_compressed_bgr = cv2.imread(file_path_comp)
                if is_color:
                    img_compressed_cv = cv2.cvtColor(img_compressed_bgr, cv2.COLOR_BGR2RGB)
                else:
                    img_compressed_cv = cv2.imread(file_path_comp, cv2.IMREAD_GRAYSCALE)
                
                # Calculate metrics
                psnr_value, ssim_value = self.calculate_image_metrics(img_original_cv, img_compressed_cv)
                
                results.append({
                    'Method': label,
                    'Type': config['type'],
                    'FileSize (KB)': compressed_size_bytes / 1024,
                    'CompressionRatio': original_size_bytes / compressed_size_bytes,
                    'PSNR (dB)': psnr_value,
                    'SSIM': ssim_value,
                    'Color': config['color']
                })
            
            # Create comprehensive comparison visualization
            fig = plt.figure(figsize=(16, 12))
            
            # File size comparison
            plt.subplot(2, 2, 1)
            methods = [r['Method'] for r in results]
            file_sizes = [r['FileSize (KB)'] for r in results]
            colors = [r['Color'] for r in results]
            
            bars = plt.bar(methods, file_sizes, color=colors, alpha=0.7)
            plt.xlabel('Compression Method')
            plt.ylabel('File Size (KB)')
            plt.title('File Size Comparison')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, size in zip(bars, file_sizes):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{size:.1f}', ha='center', va='bottom', fontsize=9)
            
            # PSNR comparison
            plt.subplot(2, 2, 2)
            psnr_values = [r['PSNR (dB)'] for r in results]
            bars = plt.bar(methods, psnr_values, color=colors, alpha=0.7)
            plt.xlabel('Compression Method')
            plt.ylabel('PSNR (dB)')
            plt.title('PSNR Comparison')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            # Compression ratio vs PSNR scatter plot
            plt.subplot(2, 2, 3)
            comp_ratios = [r['CompressionRatio'] for r in results]
            for i, result in enumerate(results):
                plt.scatter(result['CompressionRatio'], result['PSNR (dB)'], 
                           color=result['Color'], s=100, alpha=0.7, label=result['Method'])
            plt.xlabel('Compression Ratio')
            plt.ylabel('PSNR (dB)')
            plt.title('Compression Efficiency')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            
            # SSIM comparison
            plt.subplot(2, 2, 4)
            ssim_values = [r['SSIM'] if r['SSIM'] is not None else 0 for r in results]
            bars = plt.bar(methods, ssim_values, color=colors, alpha=0.7)
            plt.xlabel('Compression Method')
            plt.ylabel('SSIM')
            plt.title('SSIM Comparison')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            plt.suptitle("JPEG vs PNG Compression Comparison", fontsize=16)
            plt.tight_layout()
            
            # Convert to base64
            plot_url = self.save_plot_as_base64(fig)
            
            # Clean up
            os.remove(file_path)
            for config in test_configs:
                if config['type'] == 'JPEG':
                    file_path_comp = os.path.join(self.results_path, f'compare_jpeg_q{config["quality"]}.jpg')
                else:
                    file_path_comp = os.path.join(self.results_path, f'compare_png_l{config["level"]}.png')
                if os.path.exists(file_path_comp):
                    os.remove(file_path_comp)
            
            return self.templates.TemplateResponse("modules/modul6/result.html", {
                "request": request,
                "title": "Compression Methods Comparison",
                "plot_url": plot_url,
                "analysis_type": "Compression Comparison",
                "results": results,
                "original_size": round(original_size_bytes / 1024, 2),
                "success": True
            })
            
        except Exception as e:
            return self.templates.TemplateResponse("modules/modul6/result.html", {
                "request": request,
                "title": "Error",                "error": str(e),
                "success": False
            })
    
    async def demo_compression(self, request):
        """Demo compression analysis dengan dataset bawaan"""
        try:
            demo_images = ["cameraman.png", "hurufA.png", "tier1.png"]
            demo_results = []
            
            for img_name in demo_images:
                img_path = os.path.join(self.dataset_path, img_name)
                if os.path.exists(img_path):
                    # Load image
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    
                    original_size = os.path.getsize(img_path)
                    
                    # Test JPEG Q75
                    jpeg_path = os.path.join(self.results_path, f'demo_{img_name}_jpeg.jpg')
                    cv2.imwrite(jpeg_path, img, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    jpeg_size = os.path.getsize(jpeg_path)
                    
                    # Test PNG L6
                    png_path = os.path.join(self.results_path, f'demo_{img_name}_png.png')
                    cv2.imwrite(png_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 6])
                    png_size = os.path.getsize(png_path)
                    
                    demo_results.append({
                        "image": img_name,
                        "original_size": round(original_size / 1024, 2),
                        "jpeg_size": round(jpeg_size / 1024, 2),
                        "png_size": round(png_size / 1024, 2),
                        "jpeg_ratio": round(original_size / jpeg_size, 2),
                        "png_ratio": round(original_size / png_size, 2)
                    })
                    
                    # Clean up demo files
                    if os.path.exists(jpeg_path):
                        os.remove(jpeg_path)
                    if os.path.exists(png_path):
                        os.remove(png_path)
            
            return self.templates.TemplateResponse("modules/modul6/demo.html", {
                "request": request,
                "title": "Compression Demo Results",
                "demo_results": demo_results,
                "success": True
            })
            
        except Exception as e:
            return self.templates.TemplateResponse("modules/modul6/demo.html", {
                "request": request,
                "title": "Demo Error",
                "error": str(e),
                "success": False
            })