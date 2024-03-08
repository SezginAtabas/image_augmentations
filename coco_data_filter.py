import os
import cv2
import shutil
from pycocotools.coco import COCO





def filter_by_category(dataset, category_id, images_folder):
    """ Filters a coco dataset so that only target category remains.

    Args:
        dataset: a COCO dataset.
        category_id: target category id.
        images_folder: folder to the images

    Returns:
        A dict containing 'images', 'bboxes' and 'class_labels'
    """
    # Initialize a dictionary to store filtered data
    # The keys will be image paths, and the values will be lists of tuples
    # Each tuple contains a bounding box and its corresponding category_id
    filtered_data = {}

    # Iterate over all annotations
    for ann_id, ann in dataset.anns.items():
        # Check if the annotation's category ID matches the input category ID
        if ann['category_id'] == category_id:
            # Get the image ID associated with the annotation
            img_id = ann['image_id']
            # Get the image information
            img = dataset.loadImgs(img_id)[0]
            # Get the image path
            img_path = os.path.join(images_folder, img['file_name'])
            
            # Check if the image path is already in the dictionary
            if img_path not in filtered_data:
                # If not, add the image path with an empty list for bounding boxes and category_ids
                filtered_data[img_path] = []
            
            # Append the bounding box and its category_id to the list for this image
            filtered_data[img_path].append((ann['bbox'], category_id))

    # Convert the dictionary to the desired output format
    images = list(filtered_data.keys())
    bboxes_and_labels = list(filtered_data.values())
    
    # Extract bounding boxes and class_labels from the list of tuples
    bboxes = [[item[0] for item in bboxes_and_label] for bboxes_and_label in bboxes_and_labels]
    class_labels = [[item[1] for item in bboxes_and_label] for bboxes_and_label in bboxes_and_labels]

    # Return the filtered data as a dictionary
    return {
        'images': images,
        'bboxes': bboxes,
        'class_labels': class_labels
    }

def save_data(save_dir, data, is_path = False, name="augmented_img_", img_extension='.png'):
    """ Saves the images and annotations.

    Args:
        save_dir: dir to save.
        data: dict containing 'images', 'class_labels', 'bboxes'.
        is_path: If the image data is a path to the image or not.
        name: Name of the saved images and annonations  Defaults to "augmented_img_".
        img_extension: extension of the saved image. Defaults to '.png'.
    """
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

def main():
    
    root = ""
    ann_file = os.path.join(root, "coco_annotations.json")
    images_folder = os.path.join(root, "images")
    out_path = ""
    
    filtered_data = filter_by_category(dataset=COCO(ann_file), category_id = 1, images_folder=root) 

    save_data(out_path, filtered_data, is_path=True, name="blender_gen_1_")
if __name__ == "__main__":
    main()



