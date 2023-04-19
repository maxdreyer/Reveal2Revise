from PIL import ImageDraw, Image
import torchvision.transforms as T
import numpy as np
import torch

def insert_artifact(img, artifact_type, **kwargs):
    if artifact_type == "ch_text":
        return insert_artifact_ch_text(img, **kwargs)
    else:
        raise ValueError(f"Unknown artifact_type: {artifact_type}")
    

def insert_artifact_ch_text(img, **kwargs):

    text = kwargs.get("text", "Clever Hans")
    fill = kwargs.get("fill", (0,0,0))
    img_size = kwargs.get("img_size", 224)

    padding = 10
    
    # Random position
    end_x = img_size - 80
    end_y = img_size - 20
    valid_positions = np.array([[5,5], [5,end_y], [end_x,5], [end_x,end_y]])
    pos = valid_positions[np.random.choice(len(valid_positions))]
    pos += np.random.normal(0, 2, 2).astype(int)
    pos[0] = np.clip(pos[0], padding, end_x-padding)
    pos[1] = np.clip(pos[1], padding, end_y-padding)
    pos = tuple(pos)
    
    # Add Random Noise to color
    fill = tuple(np.clip(np.array(fill) + np.random.normal(0, 10, 3), 0, 255).astype(int))
    
    # Random size
    size_text_img = np.random.choice(np.arange(img_size-25, img_size+25))
    
    # Random Rotation
    rotation = np.random.choice(np.arange(-3,3))
    image_text = Image.new('RGBA', (size_text_img,size_text_img), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image_text)
    draw.text(pos, text=text, fill=(255,255,255))
    image_text = T.Resize((img_size, img_size))(image_text.rotate(rotation))
    
    # Insert text into image
    out = Image.composite(image_text, img, image_text)

    mask = torch.zeros((img_size, img_size))
    mask_coord = image_text.getbbox()
    mask[mask_coord[1]:mask_coord[3], mask_coord[0]:mask_coord[2]] = 1

    return out, mask