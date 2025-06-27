# ==========================================================================
# Eses Image Lens Effects
# ==========================================================================
#
# Description:
# The 'Eses Image Lens Effects' node applies a simulation of
# lens distortion and chromatic aberration to an image.
#
# This node achieves the effect by:
# 1. Defining a master barrel or pincushion distortion strength.
# 2. Applying channel-specific offsets (aberrations) to the master distortion.
# 3. Applying the final, combined distortion to each R, G, B channel independently.
# 4. Recombining the distorted channels to create the final image.
#
# Key Features:
#
# - Master Lens Distortion:
#   - k1_master_distortion: Controls the primary strength and direction
#     (barrel/pincushion) of distortion for the entire image.
#
# - Channel-Specific Aberration:
#   - k1_red_aberration, k1_green_aberration, k1_blue_aberration: Act as deltas
#     or offsets from the master distortion, creating color fringing.
#
# - Distortion Profile Control:
#   - radial_exponent: Adjusts how quickly the distortion effect ramps up
#     from the image center towards the edges.
#
# - Interpolation Quality:
#   - interpolation_mode: Selects the resampling method for smooth results.
#
# - Edge Handling:
#   - fill_mode: Determines how new pixels at the boundaries are handled.
#
# Usage:
# Set the overall lens distortion with 'k1_master_distortion'. Then, use the
# 'k1_..._aberration' sliders to introduce color fringing on top of the base
# distortion.
#
# Version: 1.0.0
#
# License: See LICENSE.txt
#
# ==========================================================================

import numpy as np
from PIL import Image
from skimage.transform import warp # type: ignore
import torch

class EsesImageLensEffects:
    """
    A ComfyUI custom node to apply lens distortion and chromatic aberration.
    """

    @classmethod
    def INPUT_TYPES(s):
        interpolation_modes = {
            "Nearest-Neighbor": 0,
            "Bilinear": 1,
            "Bicubic": 3,
            "Biquartic": 4,
            "Biquintic": 5,
        }
        
        interpolation_mode_names = list(interpolation_modes.keys())

        return {
            "required": {
                "image": ("IMAGE",),

                # NEW: Master Lens Distortion Parameter
                "k1_master_distortion": ("FLOAT", {
                    "default": 0.0,
                    "min": -2,
                    "max": 0.5,
                    "step": 0.001,
                    "round": 0.001,
                    "display": "number"
                }),

                # Radial Distortion Parameters
                "k1_red_aberration": ("FLOAT", {
                    "default": -0.01,
                    "min": -0.1,
                    "max": 0.1,
                    "step": 0.001,
                    "round": 0.001,
                    "display": "number"
                }),
                "k1_green_aberration": ("FLOAT", {
                    "default": 0.0,
                    "min": -0.1,
                    "max": 0.1,
                    "step": 0.001,
                    "round": 0.001,
                    "display": "number"
                }),
                "k1_blue_aberration": ("FLOAT", {
                    "default": 0.01,
                    "min": -0.1,
                    "max": 0.1,
                    "step": 0.001,
                    "round": 0.001,
                    "display": "number"
                }),
                "radial_exponent": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.5,
                    "max": 4.0,
                    "step": 0.01,
                    "round": 0.01,
                    "display": "number"
                }),

                # Channel Offset Parameters
                "offset_x_red": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.1, "round": 0.1, "display": "number"}),
                "offset_y_red": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.1, "round": 0.1, "display": "number"}),
                "offset_x_green": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.1, "round": 0.1, "display": "number"}),
                "offset_y_green": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.1, "round": 0.1, "display": "number"}),
                "offset_x_blue": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.1, "round": 0.1, "display": "number"}),
                "offset_y_blue": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.1, "round": 0.1, "display": "number"}),
                
                # Post-process scaling parameter
                "post_process_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 1.0,
                    "max": 3.0,
                    "step": 0.001,
                    "round": 0.001,
                    "display": "number"
                }),
                
                # Vignette Parameters
                "vignette_amount": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001, "round": 0.001, "display": "number"}),
                "vignette_size": ("FLOAT", {"default": 0.0, "min": -1, "max": 1, "step": 0.001, "round": 0.001, "display": "number"}),
                "vignette_falloff": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 8.0, "step": 0.01, "round": 0.01, "display": "number"}),
                "fit_vignette_to_frame": (["Off", "On"], {"default": "Off"}),

                # General Interpolation/Fill Parameters
                "interpolation_mode": (interpolation_mode_names, {"default": "Bilinear"}),
                "fill_mode": (["constant", "edge", "symmetric", "reflect", "wrap"], {"default": "constant"}),
                "channel_fill_color": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01, "display": "number"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_effects"
    CATEGORY = "Eses Nodes/Image Adjustments"
    TITLE = "Eses Lens Effects"

    # A dedicated internal function to 
    # handle the post-process scaling.
    def _perform_scaling(self, image_tensor, scale_factor):
        if scale_factor == 1.0:
            return image_tensor

        B, H, W, C = image_tensor.shape
        
        # Calculate the new dimensions for the initial crop
        new_W = int(W / scale_factor)
        new_H = int(H / scale_factor)
        
        # Find the top-left corner of the crop box to keep it centered
        left = (W - new_W) // 2
        top = (H - new_H) // 2
        
        # Crop the image
        cropped_img = image_tensor[:, top:top+new_H, left:left+new_W, :]
        
        # Resize the cropped image back to the original dimensions
        # PyTorch interpolate needs shape (B, C, H, W)
        img_bchw = cropped_img.permute(0, 3, 1, 2)
        
        # Use 'bicubic' for good quality zoom.
        scaled_img = torch.nn.functional.interpolate(
            img_bchw,
            size=(H, W),
            mode='bicubic',
            align_corners=True
        )
        
        # (B, H, W, C)
        return scaled_img.permute(0, 2, 3, 1)


    def _apply_vignette(self, image_tensor, amount, size, falloff, fit_to_frame):
        if amount == 0.0:
            return image_tensor
            
        B, H, W, C = image_tensor.shape
        device = image_tensor.device

        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        Y, X = torch.meshgrid(y, x, indexing='ij')

        if fit_to_frame == "Off":
            if W > H:
                Y = Y * H / W
            elif H > W:
                X = X * W / H
        
        # 1. Define vignette radius based on size. 
        # Adding a small epsilon to avoid division by zero.
        radius = (1.0 + size) + 1e-6

        # 2. Calculate distance and scale it by the radius. 
        # Clamp to handle areas outside the circle.
        dist = (torch.sqrt(X**2 + Y**2) / radius).clamp(0, 1)

        # 3. Apply a smoothing function (Smoothstep) to the distance.
        #    This fixes the harsh edge artifact when falloff is < 1.0.
        smooth_dist = dist * dist * (3.0 - 2.0 * dist)
        
        # 4. Apply the final falloff power to the smoothed distance.
        mask = smooth_dist.pow(falloff)
        mask = mask.unsqueeze(0).unsqueeze(-1).expand_as(image_tensor)

        # Blend vignette with image
        if amount > 0:
            vignette = 1.0 - (mask * amount)
            final_image = image_tensor * vignette
        else:
            vignette = mask * abs(amount)
            final_image = image_tensor + vignette
            
        return final_image.clamp(0, 1)
    

    def barrel_distortion_func(self, xy_coords, k1, k2, offset_x, offset_y, center_x, center_y, width, height, radial_exponent):
        x = xy_coords[:, 0]
        y = xy_coords[:, 1]

        nx = (x - center_x) / (width / 2)
        ny = (y - center_y) / (height / 2)

        r_d = np.sqrt(nx**2 + ny**2)

        distortion_term = r_d**radial_exponent 
        r_u = r_d / (1 + k1 * distortion_term + k2 * distortion_term**2)

        mask_zero_r = r_d == 0
        source_nx = np.where(mask_zero_r, nx, nx * (r_u / r_d))
        source_ny = np.where(mask_zero_r, ny, ny * (r_u / r_d))

        source_x = source_nx * (width / 2) + center_x + offset_x
        source_y = source_ny * (height / 2) + center_y + offset_y

        return np.stack((source_x, source_y), axis=-1)


    def apply_effects(self, image, k1_master_distortion, k1_red_aberration, k1_green_aberration, k1_blue_aberration, 
                         radial_exponent, offset_x_red, offset_y_red, offset_x_green, offset_y_green,
                         offset_x_blue, offset_y_blue, post_process_scale, vignette_amount, vignette_size, 
                         vignette_falloff, fit_vignette_to_frame, interpolation_mode, fill_mode, channel_fill_color):
        
        interpolation_order_map = {"Nearest-Neighbor": 0, "Bilinear": 1, "Bicubic": 3, "Biquartic": 4, "Biquintic": 5}
        actual_interpolation_order = interpolation_order_map.get(interpolation_mode, 1)

        if isinstance(image, torch.Tensor):
            image_np = image.squeeze(0).cpu().numpy()
        else:
            image_np = image

        image_np = image_np.astype(np.float32)

        h, w, c = image_np.shape

        red_channel = image_np[:, :, 0]
        green_channel = image_np[:, :, 1]
        blue_channel = image_np[:, :, 2]

        center_x, center_y = w / 2, h / 2

        # Calculate the final k1 value for each channel
        final_k1_red = k1_master_distortion + k1_red_aberration
        final_k1_green = k1_master_distortion + k1_green_aberration
        final_k1_blue = k1_master_distortion + k1_blue_aberration

        warp_kwargs = {
            "output_shape": (h, w),
            "mode": fill_mode,
            "order": actual_interpolation_order,
            "cval": channel_fill_color
        }
        

        # Use the final_k1 values in the warp functions
        red_distorted = warp(red_channel,
                             lambda xy: self.barrel_distortion_func(xy, final_k1_red, 0.0, offset_x_red, offset_y_red, center_x, center_y, w, h, radial_exponent),
                             **warp_kwargs)

        green_distorted = warp(green_channel,
                               lambda xy: self.barrel_distortion_func(xy, final_k1_green, 0.0, offset_x_green, offset_y_green, center_x, center_y, w, h, radial_exponent),
                               **warp_kwargs)

        blue_distorted = warp(blue_channel,
                              lambda xy: self.barrel_distortion_func(xy, final_k1_blue, 0.0, offset_x_blue, offset_y_blue, center_x, center_y, w, h, radial_exponent),
                              **warp_kwargs)

        distorted_rgb = np.stack([red_distorted, green_distorted, blue_distorted], axis=-1)
        distorted_rgb = np.clip(distorted_rgb, 0.0, 1.0)

        output_image = torch.from_numpy(distorted_rgb).unsqueeze(0)
        
        # Perform post-process scaling as the final step
        scaled_image = self._perform_scaling(output_image, post_process_scale)

        # Apply vignette as the very last effect
        final_image = self._apply_vignette(scaled_image, vignette_amount, vignette_size, vignette_falloff, fit_vignette_to_frame)

        return (final_image,)

