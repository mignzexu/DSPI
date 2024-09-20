import sys
sys.path.append(".")
from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import os
import re
import torch
from tqdm import tqdm
import glob

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def process_images_from_folder(image_folder, text_file, output_folder, BOX_TRESHOLD=0.35, TEXT_TRESHOLD=0.25):
    # Load the model
    model = load_model("GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py", "GroundingDINO/pretrained/groundingdino_swinb_cogcoor.pth")

    # Read text descriptions from the text file
    with open(text_file, 'r') as file:
        text_lines = file.readlines()

    # Get a list of image paths sorted naturally
    image_paths = sorted([os.path.join(image_folder, image_name.strip()) for image_name in os.listdir(image_folder)], key=natural_sort_key)
    boxes_dict = {}
    logits_dict = {}
    phrases_dict = {}
    hs_dict = {}
    # Process each image in the folder
    for i, image_path in enumerate(tqdm(image_paths,desc='schedule',leave=False)):
        # if i > 1:
            # break
        # Extract the original file name without extension
        file_name = os.path.basename(image_path)
        tqdm.write(file_name)

        # Find the corresponding line in the text file
        matching_line = next((line for line in text_lines if file_name in line), None)

        if matching_line:
            
            # Split the line into image name and text description using space as separator
            image_name, text_prompt = matching_line.split(None, 1)

            # Load and process the image
            # image_source: ndarray in [0, 255]
            # image: pytorch transformed with IN moments
            image_source, image = load_image(image_path)
            boxes, logits, phrases, hs = predict(
                model=model,
                image=image,
                caption=text_prompt.strip(),  # Use stripped text description
                box_threshold=BOX_TRESHOLD,
                text_threshold=TEXT_TRESHOLD
            )
            boxes_dict[file_name] = boxes
            logits_dict[file_name] = logits
            phrases_dict[file_name] = phrases
            hs_dict[file_name] = hs

            # Annotate the image
            annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)

            # Save the annotated image with the original file name
            annotated_dir = os.path.join(output_folder, 'gdino_annotated_imgs')
            os.makedirs(annotated_dir, exist_ok=True)
            cv2.imwrite(os.path.join(annotated_dir, file_name), annotated_frame)

    label_dir = os.path.join(output_folder, 'gdino_label')
    os.makedirs(label_dir, exist_ok=True)

    torch.save(boxes_dict, os.path.join(label_dir, 'boxes.pt'))
    # torch.save(logits_dict, os.path.join(label_dir,'logits.pt'))
    torch.save(phrases_dict, os.path.join(label_dir, 'phrases.pt'))
    torch.save(hs_dict, os.path.join(label_dir, 'logits.pt'))


def gdino_dataset(dataset="CARPK", image_folder="data/CARPK/Images", output_folder="data/CARPK", BOX_TRESHOLD = 0.35, TEXT_TRESHOLD = 0.25):
    model = load_model("GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py", "pretrained/groundingdino_swinb_cogcoor.pth")
    
    imgs_path = glob.glob(os.path.join(image_folder, "*.png"))
    imgs_path = imgs_path + glob.glob(os.path.join(image_folder, "*.jpg"))
    
    boxes_dict = {}
    # logits_dict = {}
    phrases_dict = {}
    hs_dict = {}
    
    if dataset == "CARPK" or dataset == "PUCPR":
        caption = "car"  
    else:
        caption = "people"
    

    for img_path in tqdm(imgs_path, desc=dataset, leave=False):
        tqdm.write(img_path)
        file_name = os.path.basename(img_path)
        image_source, image = load_image(img_path)
        boxes, logits, phrases, hs = predict(
            model=model,
            image=image,
            caption=caption,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )

        boxes_dict[file_name] = boxes.cpu()
        # logits_dict[file_name] = logits
        phrases_dict[file_name] = phrases
        hs_dict[file_name] = hs.cpu()
    
        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        annotated_dir = os.path.join(output_folder, 'gdino_annotated_imgs')
        os.makedirs(annotated_dir, exist_ok=True)
        cv2.imwrite(os.path.join(annotated_dir, file_name), annotated_frame)

    label_dir = os.path.join(output_folder, 'gdino_label')
    os.makedirs(label_dir, exist_ok=True)

    print("start save pt file ...")
    torch.save(boxes_dict, os.path.join(label_dir, 'boxes.pt'))
    # torch.save(logits_dict, os.path.join(label_dir,'logits.pt'))
    torch.save(phrases_dict, os.path.join(label_dir, 'phrases.pt'))
    torch.save(hs_dict, os.path.join(label_dir, 'logits.pt'))
    print(f"{dataset} finished")


if __name__ == "__main__": 
    dataset = "NWPU"

    if dataset == "FSC":
        IMAGE_FOLDER = "data/FSC/images_384_VarV2"
        TEXT_FILE = "data/FSC/FSC_147/ImageClasses_FSC_147.txt"
        OUTPUT_FOLDER = "out/vis/fsc_gdino"
        process_images_from_folder(IMAGE_FOLDER, TEXT_FILE, OUTPUT_FOLDER)
    else:
        if dataset == "CARPK":
            image_folder="data/CARPK/Images" 
            output_folder="data/CARPK"
        elif dataset == "PUCPR":
            image_folder="data/PUCPR/Images" 
            output_folder="data/PUCPR"
        elif dataset == "part_A_train":
            image_folder="data/ShanghaiTech/part_A/train_data/images" 
            output_folder="data/ShanghaiTech/part_A/train_data"
        elif dataset == "part_A_test":
            image_folder="data/ShanghaiTech/part_A/test_data/images" 
            output_folder="data/ShanghaiTech/part_A/test_data"
        elif dataset == "part_B_train":
            image_folder="data/ShanghaiTech/part_B/train_data/images" 
            output_folder="data/ShanghaiTech/part_B/train_data"
        elif dataset == "part_B_test":
            image_folder="data/ShanghaiTech/part_B/test_data/images" 
            output_folder="data/ShanghaiTech/part_B/test_data"
        elif dataset == "UCF50":
            image_folder="data/UCF50" 
            output_folder="data/UCF50"
        elif dataset == "QNRF":
            image_folder="data/QNRF/Test" 
            output_folder="data/QNRF"
        elif dataset == "JHU":
            image_folder="data/JHU/test/images" 
            output_folder="data/JHU/test"
        elif dataset == "NWPU":
            image_folder="data/NWPU/test/imgs"
            output_folder="data/NWPU/test"
            
        gdino_dataset(dataset, image_folder, output_folder)
