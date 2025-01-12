import cv2
import math


def divide_image(image, output_path, temp_range, n=16):
    img = image
    output_path = output_path.split("\\")[-1]
    if img is None:
        raise ValueError("Could not read the image")

    height, width = img.shape[:2]

    grid_size = int(math.sqrt(n))

    piece_height = height // grid_size
    piece_width = width // grid_size

    pieces = []

    for i in range(grid_size):
        for j in range(grid_size):
            start_y = i * piece_height
            end_y = start_y + piece_height
            start_x = j * piece_width
            end_x = start_x + piece_width

            piece = img[start_y:end_y, start_x:end_x]
            pieces.append(piece)

    for idx, piece in enumerate(pieces):
        output_path = f'training/{output_path}_piece_{idx + 1}.TIF'
        cv2.imwrite(output_path, piece)
    with open(f"training/temperature.txt", "a") as f:
        f.write(f'{output_path} {temp_range}\n')

    return pieces
