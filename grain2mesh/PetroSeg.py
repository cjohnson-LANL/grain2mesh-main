"""
© 2025. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
"""
import cv2
import numpy as np
import os
import torch
import torch.nn as nn
from skimage import segmentation
from matplotlib import pyplot as plt
from .utils import png_to_npz, make_tqdm_callback
from scipy.ndimage import gaussian_filter
from scipy.ndimage import label
from skimage import morphology


def convolution_segmentation(image_path, exp_path):
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    print("PetroSeg - stripped to bare scripts required for our task. RGH 07072025")
    print("""
    - **Training Epochs**: Higher values will lead to fewer segments but may take more time.
    - **Image Size**: For better efficiency, upload small-sized images.
    """)


    while True:
        params = set_parameters()

        original_image, image_resized = load_and_resize_image(image_path, params)

        


        print("Starting segmentation...")

        progress_callback = make_tqdm_callback('Segmentation Progress')
        segmented_image = perform_custom_segmentation(image_resized, params, progress_callback=progress_callback)

        print("Segmentation complete.")

        if params['randomColorsFlag'] == 1:
            colored_segments = randomize_colors(segmented_image)
        else:
            colored_segments = color_segments_by_original_image(segmented_image, image_resized)

        #Save the result ---->> to then go to our post processing routine
        segmented_bgr = cv2.cvtColor(colored_segments, cv2.COLOR_RGB2BGR)
        unique_labels_len = len(np.unique(colored_segments.reshape(-1, 3), axis=0))
        output_path = f"{exp_path}/A1.5_CNN_segmented_image_{unique_labels_len}_{params['train_epoch']}epochs.png"

        plt.imshow(segmented_bgr)
        plt.title("Segmented Image")
        plt.show()
        satisfied = input("Are you satisfied with the segmentation? (y to save / n to adjust / q to quit): ").strip().lower()

        if satisfied == 'y':
            cv2.imwrite(output_path, segmented_bgr)
            segmented_npz = png_to_npz(output_path)
            print(f"Segmented image saved to: {output_path}")
            print(f"Total Number of Unique Labels = {unique_labels_len}")
            print("Saved final convolution segmentation. Moving to clean up.")
            return segmented_npz

        elif satisfied == 'q':
            print("Exiting without saving.")
            exit()

def set_parameters():
    # Collect user inputs
    train_epoch = int(input("Enter train epoch (1-200): "))
    mod_dim1 = int(input("Enter mod_dim1 (1-128) (1st and 3rd channels): "))
    mod_dim2 = int(input("Enter mod_dim2 (1-128) (2nd and last channels): "))
    min_label_num = int(input("Enter min_label_num (1-20): "))
    max_label_num = int(input("Enter max_label_num (1-200): "))
    target_size_width = int(input("Enter target size width (100-1200): "))
    target_size_height = int(input("Enter target size height (100-1200): "))
    randomColorsFlag = int(input("Enter if Random colors (1) (not=0): "))

    # Creat parameter dictionary
    params = {
        'train_epoch': train_epoch,
        'mod_dim1': mod_dim1,
        'mod_dim2': mod_dim2,
        'min_label_num': min_label_num,
        'max_label_num': max_label_num,
        'target_size': (target_size_width, target_size_height),
        'randomColorsFlag': randomColorsFlag
    }

    return params

def load_and_resize_image(image_path: str, params: dict):
    """Load image, resize it, and perform segmentation."""
    if not os.path.isfile(image_path):
        print(f"File not found: {image_path}")
        return


    image = cv2.imread(image_path)
    if image is None:
        print("Error loading image. Please check the file format or path.")
        return

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize the image using the user-defined target size
    target_size = params.get('target_size', (512, 512))
    image_resized = resize_image(image_rgb, target_size)

    return image_rgb, image_resized

def resize_image(image, size):
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def perform_custom_segmentation(image, params,progress_callback=None):
    class Args(object):
        def __init__(self, params):
            self.train_epoch = params.get('train_epoch', 2 ** 3)
            self.mod_dim1 = params.get('mod_dim1', 64)
            self.mod_dim2 = params.get('mod_dim2', 32)
            self.gpu_id = params.get('gpu_id', 0)
            self.min_label_num = params.get('min_label_num', 6)
            self.max_label_num = params.get('max_label_num', 256)

    args = Args(params)

    class MyNet(nn.Module):
        def __init__(self, inp_dim, mod_dim1, mod_dim2):
            super(MyNet, self).__init__()
            self.seq = nn.Sequential(
                nn.Conv2d(inp_dim, mod_dim1, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(mod_dim1),
                nn.ReLU(inplace=True),
                nn.Conv2d(mod_dim1, mod_dim2, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(mod_dim2),
                nn.ReLU(inplace=True),
                nn.Conv2d(mod_dim2, mod_dim1, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(mod_dim1),
                nn.ReLU(inplace=True),
                nn.Conv2d(mod_dim1, mod_dim2, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(mod_dim2),
            )

        def forward(self, x):
            return self.seq(x)

    torch.cuda.manual_seed_all(1943)
    np.random.seed(1943)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    '''segmentation ML'''
    seg_map = segmentation.felzenszwalb(image, scale=15, sigma=0.06, min_size=14)
    seg_map = seg_map.flatten()
    seg_lab = [np.where(seg_map == u_label)[0]
            for u_label in np.unique(seg_map)]
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    tensor = image.transpose((2, 0, 1))
    tensor = tensor.astype(np.float32) / 255.0
    tensor = tensor[np.newaxis, :, :, :]
    tensor = torch.from_numpy(tensor).to(device)

    model = MyNet(inp_dim=3, mod_dim1=args.mod_dim1, mod_dim2=args.mod_dim2).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-2, momentum=0.9)

    image_flatten = image.reshape((-1, 3))
    color_avg = np.random.randint(255, size=(args.max_label_num, 3))
    show = image


    for batch_idx in range(args.train_epoch):
        if progress_callback:
            progress_callback(batch_idx + 1, args.train_epoch) # added RGH 07072025

        optimizer.zero_grad()
        output = model(tensor)[0]
        output = output.permute(1, 2, 0).view(-1, args.mod_dim2)
        target = torch.argmax(output, 1)
        im_target = target.data.cpu().numpy()

        for inds in seg_lab:
            u_labels, hist = np.unique(im_target[inds], return_counts=True)
            im_target[inds] = u_labels[np.argmax(hist)]

        target = torch.from_numpy(im_target)
        target = target.to(device)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        un_label, lab_inverse = np.unique(im_target, return_inverse=True, )
        if un_label.shape[0] < args.max_label_num:
            img_flatten = image_flatten.copy()
            if len(color_avg) != un_label.shape[0]:
                color_avg = [np.mean(img_flatten[im_target == label], axis=0, dtype=int) for label in un_label]
            for lab_id, color in enumerate(color_avg):
                img_flatten[lab_inverse == lab_id] = color
            show = img_flatten.reshape(image.shape)

    return show

def randomize_colors(segmented_image: np.ndarray) -> np.ndarray:
    unique_labels = np.unique(segmented_image.reshape(-1, 3), axis=0)
    random_colors = {
        tuple(label): tuple(np.random.randint(0, 256, size=3))
        for label in unique_labels
    }

    new_image = segmented_image.copy()
    for old_color, new_color in random_colors.items():
        mask = np.all(segmented_image == np.array(old_color), axis=-1)
        new_image[mask] = new_color

    return new_image

def color_segments_by_original_image(segmented_image: np.ndarray, original_image: np.ndarray) -> np.ndarray:
    """
    Color the segmented image based on the average RGB values of each segment
    in the original image.

    Parameters:
        segmented_image (np.ndarray): A 2D label map (HxW) or a 3-channel RGB map from segmentation
        original_image (np.ndarray): The original RGB image (HxWx3)

    Returns:
        colored_image (np.ndarray): RGB image where each segment has a mean color
    """
    h, w = original_image.shape[:2]

    # If the segmentation image is RGB, reduce it to integer labels
    if segmented_image.ndim == 3:
        # Convert unique RGB colors to labels
        unique_colors, labeled = np.unique(segmented_image.reshape(-1, 3), axis=0, return_inverse=True)
        label_map = labeled.reshape(h, w)
    else:
        label_map = segmented_image

    output = np.zeros_like(original_image)
    for label in np.unique(label_map):
        mask = label_map == label
        avg_color = np.mean(original_image[mask], axis=0).astype(np.uint8)
        output[mask] = avg_color

    return output


def smooth(image, sigma):
    unique_colors = np.unique(image)
    smoothed_stack = []

    for color in unique_colors:
        mask = image == color
        smoothed = gaussian_filter(mask.astype(float), sigma=sigma)
        smoothed_stack.append(smoothed)

    smoothed_stack = np.stack(smoothed_stack, axis=0)

    max_idx = np.argmax(smoothed_stack, axis=0)
    output = unique_colors[max_idx]

    val = 1
    np.unique(output)
    for i in np.unique(output):
        output[output == i] = val
        val += 1

    return output


def label_segments(image):
    labeled = np.zeros_like(image)

    current_label = 0
    for value in np.unique(image):
        if value == 0:
            continue
        mask = (image == value)
        
        # Label connected components in the mask
        components, num_features = label(mask)

        # Add labels to the result (offset by current_label)
        labeled[components > 0] = components[components > 0] + current_label
        current_label += num_features

    return labeled.astype(int)

