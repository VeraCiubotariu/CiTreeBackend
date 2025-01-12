import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import requests
import os

class RRDBNet(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32):
        super(RRDBNet, self).__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat_first = self.conv_first(x)
        body_feat = self.body(feat_first)
        body_out = self.conv_body(body_feat)
        feat = feat_first + body_out

        # Upsampling
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch):
        super(RRDB, self).__init__()
        self.rdb1 = RDB(num_feat, num_grow_ch)
        self.rdb2 = RDB(num_feat, num_grow_ch)
        self.rdb3 = RDB(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch):
        super(RDB, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


def create_blend_mask(height, width, overlap):
    """
    Creates a blending mask for smooth tile transitions with proper edge case handling.

    Args:
        height (int): Height of the tile
        width (int): Width of the tile
        overlap (int): Size of the overlapping region

    Returns:
        numpy.ndarray: Blending mask with gradual weights
    """
    mask = np.ones((height, width), dtype=np.float32)

    # Adjust overlap to not exceed tile dimensions
    effective_overlap = min(overlap, min(height, width) // 2)

    if effective_overlap > 0:
        # Create gradual transition in overlapping regions
        for i in range(effective_overlap):
            # Calculate weight for current position (linear transition)
            weight = i / effective_overlap

            # Apply weights only if within bounds
            if i < width:
                # Left edge
                mask[:, i] *= weight
                # Right edge
                if i < width:
                    mask[:, -(i + 1)] *= weight

            if i < height:
                # Top edge
                mask[i, :] *= weight
                # Bottom edge
                mask[-(i + 1), :] *= weight

            # Handle corners (use minimum of horizontal and vertical weights)
            if i < min(height, width):
                corner_weight = weight * weight
                mask[i, i] = corner_weight
                if i < width - 1:
                    mask[i, -(i + 1)] = corner_weight
                if i < height - 1:
                    mask[-(i + 1), i] = corner_weight
                    if i < width - 1:
                        mask[-(i + 1), -(i + 1)] = corner_weight

    return mask


def load_model():
    model = RRDBNet()
    url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
    if not os.path.exists('RealESRGAN_x4plus.pth'):
        print("Downloading pre-trained model...")
        response = requests.get(url)
        with open('RealESRGAN_x4plus.pth', 'wb') as f:
            f.write(response.content)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loadnet = torch.load('RealESRGAN_x4plus.pth', map_location=device)
    if 'params_ema' in loadnet:
        keyname = 'params_ema'
    else:
        keyname = 'params'
    model.load_state_dict(loadnet[keyname], strict=True)
    model.to(device)
    return model, device


def process_image_tiles(input_image, tile_size=512, tile_overlap=64):
    """
    Process large images by splitting them into tiles, upscaling each tile,
    and then reconstructing the final image with smooth blending between tiles.

    Args:
        input_image (str): Image

        tile_size (int): Size of each tile (default: 512)
        tile_overlap (int): Overlap between tiles for blending (default: 64)
    """
    print("Loading model...")
    model, device = load_model()
    model.eval()

    print("Loading image...")
    img = input_image
    input_image = np.array(img)

    # Calculate output dimensions (3x upscaling)
    output_height = int(input_image.shape[0] * 3)
    output_width = int(input_image.shape[1] * 3)

    # Initialize output arrays with proper dimensions
    output_image = np.zeros((output_height, output_width, 3), dtype=np.float32)
    weight_accumulator = np.zeros((output_height, output_width, 1), dtype=np.float32)

    # Calculate number of tiles needed
    num_tiles_x = int(np.ceil(input_image.shape[1] / (tile_size - tile_overlap)))
    num_tiles_y = int(np.ceil(input_image.shape[0] / (tile_size - tile_overlap)))

    print(f"Processing image in {num_tiles_x * num_tiles_y} tiles...")

    for tile_y in range(num_tiles_y):
        for tile_x in range(num_tiles_x):
            # Calculate tile coordinates
            x_start = tile_x * (tile_size - tile_overlap)
            y_start = tile_y * (tile_size - tile_overlap)
            x_end = min(x_start + tile_size, input_image.shape[1])
            y_end = min(y_start + tile_size, input_image.shape[0])

            # Extract and process tile
            tile = input_image[y_start:y_end, x_start:x_end]

            actual_tile_height = y_end - y_start
            actual_tile_width = x_end - x_start

            # Skip processing if tile dimensions are too small
            if actual_tile_height < 3 or actual_tile_width < 3:
                continue

            # Create blending mask for this tile
            blend_mask = create_blend_mask(actual_tile_height, actual_tile_width, tile_overlap)

            # Prepare tile for model
            tile = tile.astype(np.float32) / 255.0
            tile = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).to(device)

            # Process tile through model
            with torch.no_grad():
                try:
                    output = model(tile)
                    target_h = int(actual_tile_height * 3)
                    target_w = int(actual_tile_width * 3)
                    output = F.interpolate(output, size=(target_h, target_w),
                                           mode='bicubic', align_corners=False)
                except RuntimeError as e:
                    print(f"Error processing tile at ({tile_x}, {tile_y}): {e}")
                    continue

            # Convert output back to numpy array
            output = output.squeeze().permute(1, 2, 0).clamp_(0, 1).cpu().numpy()

            # Calculate output tile position
            out_x_start = x_start * 3
            out_y_start = y_start * 3
            out_x_end = out_x_start + output.shape[1]
            out_y_end = out_y_start + output.shape[0]

            # Upscale the blending mask
            blend_mask_upscaled = cv2.resize(blend_mask, (output.shape[1], output.shape[0]),
                                             interpolation=cv2.INTER_LINEAR)
            blend_mask_upscaled = blend_mask_upscaled[..., np.newaxis]

            # Apply blending mask and accumulate
            output_image[out_y_start:out_y_end, out_x_start:out_x_end] += output * blend_mask_upscaled
            weight_accumulator[out_y_start:out_y_end, out_x_start:out_x_end] += blend_mask_upscaled

            print(f"Processed tile {tile_y * num_tiles_x + tile_x + 1}/{num_tiles_x * num_tiles_y}")

    # Normalize by accumulated weights
    output_image = np.divide(output_image, weight_accumulator,
                             out=np.zeros_like(output_image),
                             where=weight_accumulator != 0)

    # Convert to uint8 for saving
    output_image = (output_image * 255.0).round().astype(np.uint8)

    print("Saving result...")
    return Image.fromarray(output_image)
    print("Done!")
