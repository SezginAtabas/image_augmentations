import os
import cv2
import torch
import shutil
import random

import albumentations as A
from albumentations.core import convert_bbox_to_albumentations, convert_bboxes_from_albumentations, convert_bboxes_to_albumentations

from ultralytics.utils.ops import clip_boxes

def load_data(root):
    
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
     
    print(f"loaded { len(data['images']) }")   
    return data

def convert_to_albumentations(images, bboxes, clip = True):
    assert len(images) == len(bboxes)
    
    
    bboxes_out = []
    to_add = []
    for idx, boxes in enumerate(bboxes):
        
        # get the image size for the conversion
        img_height, img_width = cv2.imread(images[idx]).shape[:2]
        
        to_add = []
        for box in boxes: 
            # convert each box to albumentations
            converted_bbox = convert_bbox_to_albumentations(bbox=box, source_format='yolo', rows=img_height, cols=img_width)

            converted_bbox = [round(e) if e < 0 else e for e in converted_bbox]
            
            
            to_add.append(converted_bbox)
        
        bboxes_out.append(to_add)
        
    return bboxes_out

def save_data(save_dir, data, is_path = False, name="augmented_img_", img_extension='.png'):
    
    
    img_save_dir = os.path.join(save_dir, 'images')
    label_save_dir = os.path.join(save_dir, 'labels')

    # make sure folders exist
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(img_save_dir, exist_ok=True)
    os.makedirs(label_save_dir, exist_ok=True)

   
    for idx, (img_data, bboxes, class_labels) in enumerate(zip(data['images'], data['bboxes'], data['class_labels'])):
        # save the image
        img_save_path = os.path.join(img_save_dir, name + str(idx) + img_extension)
        
        # if image data is path to the image
        if is_path:
            # move the img using shutil
            shutil.copy(img_data, img_save_path)
            
        else:
            # save the img
            cv2.imwrite(img_save_path, img_data)  
        
        
        # save the labels
        label_save_path = os.path.join(label_save_dir, name + str(idx) + '.txt')
        with open(label_save_path, 'w') as file:
            for box, label in zip(bboxes, class_labels):
                formatted_box = " ".join(f"{coord:.10f}" for coord in box)
                file.write(f"{label} {formatted_box}\n")

    print(f"saved {len(data['images'])} images to {img_save_dir}")
    print(f"saved {len(data['images'])} labels to {label_save_dir}")

def augment(data, transform=None, p=1.0, augment_save_dir = None, aug_save_name = "augment_data.yaml"):   
        
    transformed_data = {'images' : [], 'bboxes' : [], 'class_labels' : []}
    skipped = 0
        
    # loop over all images
    for idx, (im_path, bbox, class_label) in enumerate(zip(data['images'], data['bboxes'], data['class_labels'])):
        if p < random.random():
            continue
            
        # read img data
        im = cv2.imread(im_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        image_height, image_width = im.shape[:2]
        
        
        try:
            # transform the image
            data = transform(image=im, bboxes=bbox, class_labels=class_label)
        
        except Exception as e:
            print(e)
            skipped +=1
            continue
        
        
        # bboxes might get deleted because of transforms, skip them.
        if data['bboxes']:
            
            transformed_data['images'].append(data['image'])
            transformed_data['class_labels'].append(data['class_labels'])
            transformed_data['bboxes'].append(data['bboxes'])
                    
        else:
            skipped += 1
            print("No bbox found after augmentation skipping image.")
            
            
    print(f"augmented {len(transformed_data['images'])} images. skipped {skipped} images")
        
    # save the list augmentations
    if augment_save_dir:  
        os.makedirs(augment_save_dir, exist_ok=True)
        save_path = os.path.join(augment_save_dir, aug_save_name)  
            
        print(f"saving transform to {save_path}")
        A.save(transform=transform, filepath=save_path, data_format='yaml',on_not_implemented_error='warn')
            
        return transformed_data


def visualize(bboxes, img, convert):
    img_height, img_width = img.shape[:2]   
        
    if convert:
        bboxes = convert_bboxes_to_albumentations(bboxes, source_format='yolo', rows=img_height, cols=img_width)
    
    for bbox in bboxes:
           
        x_min, y_min, x_max, y_max = bbox
        
        x_min = int(x_min * img_width)
        x_max = int(x_max * img_width)
        y_min = int(y_min * img_height)
        y_max = int(y_max * img_height)
        
            
        c1, c2 = (x_min, y_min), (x_max, y_max)
            
        cv2.rectangle(img=img, pt1=c1, pt2=c2, color=(255, 0, 0), thickness=4, lineType=cv2.LINE_AA)




transform = A.Compose([
    
    A.NoOp()
    #A.Resize(height=640, width=640, interpolation=cv2.INTER_LANCZOS4) 
    
    #A.HorizontalFlip(),
    #A.ColorJitter(contrast=0.1),
    #A.BBoxSafeRandomCrop(),
    #A.Flip(),
    #A.Rotate(interpolation=cv2.INTER_LANCZOS4),
    #A.GaussNoise(var_limit=(5.0, 15.0)),
    #A.GaussianBlur(blur_limit=1, p=0.3),
    #A.RandomScale(scale_limit=(-0.25, 0.15), interpolation=cv2.INTER_LANCZOS4, p=0.45),
    #A.Cutout(num_holes=5, max_h_size=40, max_w_size=40, p=0.4)
    
    
    
    #A.Transpose(p=0.4),
    #A.Flip(p=0.5),
    #A.RandomScale(scale_limit=(-0.25, 0.15), interpolation=cv2.INTER_LANCZOS4, p=0.45),
    #A.Affine(scale=((0.95, 1.15)), translate_percent=(0.3, 0.02), shear=(-10, 10), interpolation=cv2.INTER_LANCZOS4, p=0.8),                                                 
    #A.Perspective(scale=(0.015, 0.02), keep_size=False, interpolation=cv2.INTER_LANCZOS4, p=0.25),
    #A.PadIfNeeded(min_height=720, min_width=1280, border_mode=cv2.BORDER_CONSTANT, position=A.PadIfNeeded.PositionType.RANDOM),
                                           
], bbox_params=A.BboxParams(format='albumentations', min_visibility=0.5, label_fields=['class_labels']))


def main():
    data_dir = ""
    out_dir = ""
    
    data = load_data(data_dir)
    
    aug_data = augment(data, transform, p=1.0, augment_save_dir=out_dir, aug_save_name="data_aug_train_1#final##" )
    
    # convert to yolo format for training
    aug_data['bboxes'] = [convert_bboxes_from_albumentations(tuple(map(tuple, d)), target_format='yolo', 
                                rows=640, cols=640, check_validity=True) for d in aug_data['bboxes']]
    
    save_data(out_dir, aug_data, is_path=False, name="data_aug_train_1#final#")
    

    idx = 0
    while True:
        # convert if bbox format is not in albumentations.
        visualize(aug_data['bboxes'][idx], aug_data['images'][idx], convert=True)
        cv2.imshow('img', aug_data['images'][idx])
        
        key = cv2.waitKey(0)
        
        if key == ord('q'):
            break
        
        if key == ord('d') and idx < len(aug_data['bboxes']) - 1:
            idx +=1
            
        if key == ord('a') and idx > 0:
            idx -=1    
            
    
    
    cv2.destroyAllWindows()
    
    

if __name__ == "__main__":
    main()







