import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset

from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F
from torchvision.utils import draw_bounding_boxes
from torchvision.ops import clip_boxes_to_image, box_convert
from torchvision.io import read_image, write_file





class CustomVisionDataset(Dataset):
    def __init__(self, root, transform):
        
        self.data = self.load_data(root)
        self.transform = transform
    
    def __len__(self):
        return len(self.data['images'])
    
    def __getitem__(self, idx):
        
        # read the image
        img = read_image(self.data['images'][idx])
        # get the bounding boxes 
        boxes = self.data['bboxes'][idx]
        # get the class labels
        class_labels = self.data['class_labels'][idx]
        
        # convert img to tv_tensor
        img = tv_tensors.Image(img)
        
        # wrap the targets into tv_tensor
        target = {}
        target['boxes'] = tv_tensors.BoundingBoxes(boxes, format='XYXY', canvas_size= F.get_size(img), dtype=torch.float32)
        target['labels'] = torch.tensor(class_labels, dtype=torch.int64)
        
        if self.transform is not None:
            img, target = self.transform(img, target)
            
        return img, target
    
    def label_read_helper(self, str_label : str):
        if '.' in str_label:
            s = str_label.split('.') 
            return float(s[0] + s[1])
        else:
            return float(str_label)
        
    
    def load_data(self, root):
        """ Reads images and labels from root folder. Returns a dict containing images, class_labels and bboxes
        Assumes images are under root/images and labels are in root/labels And have the same filename.
        
        Args:
            root: folder to read images and labels from 

        Returns:
            data: A python dict containing images, bboxes and class_labels. keys 'images', 'bboxes', 'class_labels'
        """
    
        img_dir = os.path.join(root, "images") 
        label_dir = os.path.join(root, "labels")   
        
        img_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith((".jpg", ".png"))])
        label_paths = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith(".txt")])

        assert len(img_paths) == len(label_paths)
        
        data = {"images" : img_paths, "bboxes" : [], "class_labels" : []}
        
        for idx, label_path in enumerate(label_paths):  
            
            to_add_box = []
            to_add_label = []
            # read the bounding box and label data
            with open(label_path, "r") as file:     
                
                for line in file:
                    d = list(map(float, line.strip().split(" ")))  
                    
                    to_add_box.append(d[1 :])
                    to_add_label.append(d[0])
            
            # add the bounding box and label data to a dict
            data["bboxes"].append(to_add_box)
            data["class_labels"].append(to_add_label)
            
        assert len(data["images"]) == len(data["bboxes"]) == len(data["class_labels"])    
        
        print(f"loaded { len(data['images']) } images")   
        return data


def collate_fn(data):
    images, targets = [], []
    for item in data:
        if len(item) == 2:
            image, target = item
            images.append(image)
            targets.append(target)
            
        elif len(item) == 1:
            print("warning image without target!")
            # Handle single-value element (e.g., image without target)
            image = item[0]
            images.append(image)
            targets.append(None)  # Placeholder for missing target
        
        else:
            raise ValueError("Invalid data format: expected 1 or 2 values per item")
        
    return images, targets


def save_from_dataloader(dataloader : DataLoader, save_dir, save_name = "img_"):
    
    # path to save the images and targets
    img_save_dir = os.path.join(save_dir, 'images')
    target_save_dir = os.path.join(save_dir, 'labels')
    
    # create the needed dirs if they dont exist
    os.makedirs(img_save_dir, exist_ok=True)
    os.makedirs(target_save_dir, exist_ok=True)
    
    
    idx = 0
    for batch_images, batch_targets in dataloader:
        
        for image, target in zip(batch_images, batch_targets):
            # convert to PIL image
            img = F.to_pil_image(image)
            img_save_path = os.path.join(img_save_dir, save_name + str(idx) + '.png')
            img.save(img_save_path)

            # save the bounding boxes and labels
            target_save_path = os.path.join(target_save_dir, save_name + str(idx) + '.txt')

            # target is already on cpu unless device is specified on tv_tensor
            with open(target_save_path, 'w') as file:    
                for bbox, class_label in zip(target['boxes'], target['labels']):
                    file.write(f"{class_label} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
            
            idx +=1
    
    print(f"Saved {idx} images to {img_save_dir}")
    print(f"Saved {idx} labels to {target_save_dir}")         
                
                
def augment_cycle(transforms, save_dir, input_dir):
    #NOTE: VERY inefficient does unnecessary reading and saving.
    
    
    print(f"starting augmentation cycle for {len(transforms)} cycles.")
    
    paths_to_read = [input_dir, ]
    for idx, transform in enumerate(transforms):
        
        all_datasets = []
        
        # read all the datasets
        for dataset_path in paths_to_read:
            all_datasets.append(CustomVisionDataset(root=dataset_path, transform=transform))
        
        # create a dataloader using concated dataset
        final_dataloader = DataLoader(ConcatDataset(all_datasets), batch_size=16, num_workers=4, shuffle=True, collate_fn=collate_fn)
        
        # create a folder to save the new dataset
        new_save_dir = os.path.join(save_dir, 'aug_cycle' + str(idx))
        os.makedirs(new_save_dir, exist_ok=True)
        
        # save the dataset
        save_from_dataloader(final_dataloader, save_dir=new_save_dir, save_name='cycle_' + str(idx) + '_img_')
            
        # add the saved dataset to paths to read
        paths_to_read.append(new_save_dir)
    

train_transform_1 = v2.Compose([
    v2.RandomOrder
    ([
        v2.RandomRotation(degrees=(15), interpolation=v2.InterpolationMode.BILINEAR),
        v2.ColorJitter(brightness=0.15, hue=0.1, contrast=0.1, saturation=0.1),
        v2.RandomAdjustSharpness(sharpness_factor=1.1, p=0.3),
        v2.RandomHorizontalFlip(p=0.5),     
    ]),
    v2.ClampBoundingBoxes(),
    v2.SanitizeBoundingBoxes()
])
    
train_transform_2 = v2.Compose([
    v2.RandomOrder
    ([
        v2.RandomResizedCrop(size=(640, 640), interpolation=v2.InterpolationMode.BILINEAR),
        v2.RandomHorizontalFlip(),
        v2.RandomZoomOut(p=0.45),
    ]),
    v2.ClampBoundingBoxes(),
    v2.SanitizeBoundingBoxes()  
])

train_transform_3 = v2.Compose([
    v2.RandomAffine(degrees=7, translate=(0.1, 0.05), shear=15, interpolation=v2.InterpolationMode.BILINEAR),
    v2.RandomApply([
        v2.GaussianBlur(kernel_size=3),
    ], p=0.2),
    v2.RandomGrayscale(p=0.1),
    
    v2.Resize(size=(640, 640), interpolation=v2.InterpolationMode.BILINEAR),
    v2.ClampBoundingBoxes(),
    v2.SanitizeBoundingBoxes(), 
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.37953120470046997], std=[0.2255478948354721])
])    
 
 
trf = v2.Compose([
    v2.ToDtype(dtype=torch.float32, scale=True),
    v2.Normalize(mean=[0.37953120470046997], std=[0.2255478948354721])
])
 
val_t = v2.Compose([
    v2.Resize(size=640)
])
 
 
def get_mean_std(dataloader):
    # Initialize variables to accumulate mean and std
    mean =  0.0
    std =  0.0
    num_samples =  0.0

    for batch in dataloader:
        # Assuming each batch is a list of images and targets
        images, _ = batch
        # Convert list of images to a tensor
        images = torch.stack(images)
        # Calculate the mean and std for each image in the batch
        batch_mean = images.mean(dim=(1,  2,  3))
        batch_std = images.std(dim=(1,  2,  3))
        # Accumulate the mean and std
        mean += batch_mean.sum()
        std += batch_std.sum()
        num_samples += images.size(0)

    # Calculate the overall mean and std
    mean /= num_samples
    std /= num_samples

    return mean, std

 
def main():
    
    # mean: 0.37953120470046997
    # std: 0.2255478948354721

    
    input_dir = ''
    output_dir = ''
    
    train_dataset = CustomVisionDataset(root=input_dir, transform=val_t)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True, num_workers=4, collate_fn = collate_fn )
    
    save_from_dataloader(dataloader=train_dataloader, save_dir=output_dir)
    
    """
    input_dir = ''
    output_dir = ''
    
    transforms = [train_transform_1, train_transform_2, train_transform_3]     
    augment_cycle(transforms, output_dir, input_dir) 
   """
   

if __name__ == '__main__':
    main()
    
    
    
    










