import os
import numpy as np
import shutil
import cv2
import xml.etree.ElementTree as ET



ANOT_DIR = "data/things/things2/"
IMG_EXT  = "jpg"
NAMES_LIST = [ "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", 
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", 
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", 
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", 
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", 
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", 
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", 
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", 
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
IMG_DIR = ""
NAME = "things3"

OUT_PATH = "data/coco_objects_improved/lables"
OUT_PATH_FILES = "data/coco_objects_improved/split_files"


def yolo_to_pascalvoc(yolo_dir, output_dir, img_size, classes):
    """
    Convert YOLO format label files in a directory to Pascal VOC format.

    Args:
        yolo_label_dir (str): Directory containing YOLO format label files.
        image_size (tuple): Tuple containing image width and height (e.g., (640, 480)).
        classes (list): List of class names.

    Returns:
        pascal_voc_labels (dict): Dictionary containing Pascal VOC format labels for each image.
    """
    cwd = os.getcwd()
    yolo_dir = os.path.join(cwd, yolo_dir)
    output_dir = os.path.join(cwd, output_dir)

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    out_labels = os.path.join(output_dir, 'xml_labels')
    out_images = os.path.join(output_dir, 'xml_images')
    if not os.path.isdir(out_labels):
        os.mkdir(out_labels)
    if not os.path.isdir(out_images):
        os.mkdir(out_images)

    pascal_voc_labels = {}

    yolo_label_dir = os.path.join(yolo_dir, 'labels')
    yolo_image_dir = os.path.join(yolo_dir, 'images')

    for filename in os.listdir(yolo_label_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(yolo_label_dir, filename), "r") as file:
                lines = file.readlines()

            image_dir = os.path.join(yolo_image_dir, filename[:-3] + 'jpg')
            img = cv2.imread(image_dir)
            image_width, image_height = img.shape[:2]
            scale_width = img_size[0]/image_width
            scale_height = img_size[1]/image_height
            img = cv2.resize(img, img_size)
            root = ET.Element("annotation")
            ET.SubElement(root, "folder").text
            ET.SubElement(root, "filename").text = filename.replace(".txt", ".jpg")
            ET.SubElement(root, "path").text = filename.replace(".txt", ".jpg")
            
            size = ET.SubElement(root, "size")
            ET.SubElement(size, "width").text = str(img_size[0])
            ET.SubElement(size, "height").text = str(img_size[1])
            ET.SubElement(size, "depth").text = "3"  # Assuming RGB images

            ET.SubElement(root, "segmented").text = str(0)

            for line in lines:
                class_id, x_center, y_center, width, height = map(float, line.split())

                # Calculate bounding box coordinates
                # xmin = int((x_center - width / 2) * image_width)
                # xmax = int((x_center + width / 2) * image_width)
                # ymin = int((y_center - height / 2) * image_height)
                # ymax = int((y_center + height / 2) * image_height)

                xmin = max(int((x_center - width / 2) * img_size[0]),0)
                xmax = min(int((x_center + width / 2) * img_size[0]), img_size[0])
                ymin = max(int((y_center - height / 2) * img_size[1]), 0)
                ymax = min(int((y_center + height / 2) * img_size[1]), img_size[1])

                object_elem = ET.SubElement(root, "object")
                ET.SubElement(object_elem, "name").text = classes[int(class_id)]
                ET.SubElement(object_elem, "pose").text = "Unspecified"
                ET.SubElement(object_elem, "truncated").text = "0"
                ET.SubElement(object_elem, "difficult").text = "0"
                ET.SubElement(object_elem, "occluded").text = "0"
                bndbox = ET.SubElement(object_elem, "bndbox")
                ET.SubElement(bndbox, "xmin").text = str(xmin)
                ET.SubElement(bndbox, "xmax").text = str(xmax)
                ET.SubElement(bndbox, "ymin").text = str(ymin)
                ET.SubElement(bndbox, "ymax").text = str(ymax)

            indent(root)

            tree = ET.ElementTree(root)
            xml_filename = filename.replace(".txt", ".xml")
            tree.write(os.path.join(out_labels, xml_filename))
            image_filename = filename.replace(".txt", ".jpg")
            cv2.imwrite(os.path.join(out_images, image_filename), img)


def indent(elem, level=0):
    """
    Indent XML elements recursively for better readability.

    Args:
        elem (Element): XML element to be indented.
        level (int): Current indentation level.
    """
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def map_lable_to_class(anot_dir, lable_list):
    """This function is used to prepare the yolo data before uploading it to roboflow, this way we have an accurate mapping of the classes"""
    cwd = os.getcwd()
    anot_dir = os.path.join(cwd, anot_dir)
    if not os.path.isdir(anot_dir):
        print("wrong input path")
        exit()
    file_list = os.listdir(anot_dir)

    for file in file_list:
        current_file = os.path.join(anot_dir, file)
        open_file = open(current_file, 'r')
        lines = open_file.readlines()
        new_lines = []
        for line in lines:
            if len(line) == 1:
                break
            line = line.strip()
            new_line = line.split(' ')
            new_line[0] = lable_list[int(new_line[0])]
            new_line = ' '.join(new_line)
            new_lines.append(new_line)
        #write the new lines to the file
        write_file(current_file, new_lines)

def generate_dummy_damples(num_classes, input_image, output_dir):
    """solution to evaluation over dfferent classes when not all classes are present in the data this helps to solve the devision by zero problem
    inputs: 
        -num_classes: this var is used to set up the number of dummy samples example for coco this would be 80
        -input_image: this is the directory for the image that will be used for all the dummy annotations
        -output_dir: this directory is used to write the labels and images to. this happens to the respective folders images and lables
    outputs:
        -the output is written to files. on one had the images file gets a dummy image for every class and the lables folder gets a dummy label
        
    The lables will be written in the yolo format
    format for naming the dummy files
    dummy_{index}.jpg
    dummy_{index}.txt"""
    cwd = os.getcwd()
    input_image = os.path.join(cwd, input_image)
    output_dir = os.path.join(cwd, output_dir)

    lables_dir = os.path.join(output_dir, "labels")
    images_dir = os.path.join(output_dir, "images")

    #cheking the directorys
    if not os.path.isfile(input_image):
        print("the input image dous not exist")
        exit()

    if not os.path.isdir(output_dir):
        print("Output folder dous not exist.\n", "Making the output directory")
        os.mkdir(output_dir)
        os.mkdir(lables_dir)
        os.mkdir(images_dir)

    if not os.path.isdir(lables_dir):
        print("lables dir did not exist. This probably means that the folder you wanted to add the dummy lables to dous not exist")
        exit()
    if not os.path.isdir(images_dir):
        print("Images dir dous not exist this most likely means that the dir you wanted to add the images to dous not exist")
        exit()


    img = cv2.imread(input_image)

    #we store one image and a the annotation will contain all the classes
    cv2.imwrite(os.path.join(images_dir, "dummy" + '.jpg'), img)
    label = []
    for i in range(num_classes):
        label.append(str(i) + " 0.5 0.5 0.1 0.1")
    
    write_file(os.path.join(lables_dir, "dummy" + '.txt'), label)


#we base the split on the lables so filename reffers to the lable file names
def split_data(anot_dir, out_dir, split_ratio):
    """Function to create the needed files for yolox
    inputs:
        -anot_dir: the directory that contains the annotation files
        -out_dir: the desired output directory for the train and val .txt files
        -split_ratio: floating point to indicate the fraction of the data that is to be used for training"""
    #function to create a training and validation txt file used by the yolox repo
    #select random
    if not os.path.isdir(anot_dir):
        print("input file dous not exist")
        return 0
    
    if not os.path.isdir(out_dir):
        print("making the output dir")
        os.mkdir(out_dir)

    train_dir = os.path.join(out_dir, 'train.txt')
    valid_dir = os.path.join(out_dir, 'valid.txt')

    
    anot_list = sorted(os.listdir(anot_dir))
    num_of_elements = len(anot_list)
    index_array = np.random.choice([0, 1], size=(num_of_elements,), p=[1-split_ratio, split_ratio])

    train_list = []
    val_list = []
    for i in range(num_of_elements):
        #the index one stands for the traing set
        item = '.'.join(anot_list[i].split('.')[:-1])
        if index_array[i] == 1:
            train_list.append(item)
        else:
            val_list.append(item)

    #writing the training and validation list
    write_file(train_dir, train_list)
    write_file(valid_dir, val_list)
    
def write_file(file_path, content):
    with open(file_path, 'w') as f:
        for item in content:
            f.write(item + '\n')  # Write each item to a new line in the file


def extract_data(input_folder, output_folder):
    main_dir = os.getcwd()
    input_folder = os.path.join(main_dir, input_folder)
    output_folder = os.path.join(main_dir, output_folder)
    """This function is used to separate the xml and jpg data and place tham into lables an images folders"""
    if not os.path.isdir(input_folder):
        print("input file is not a correct directory")
    
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    lables_dir = os.path.join(output_folder, 'labels')
    images_dir = os.path.join(output_folder, 'images')

    if not os.path.isdir(lables_dir):
        os.mkdir(lables_dir)
    if not os.path.isdir(images_dir):
        os.mkdir(images_dir)

    file_list = os.listdir(input_folder)
    for item in file_list:
        file = os.path.join(input_folder, item)
        if item.split('.')[-1] == 'jpg':
            output_file = os.path.join(images_dir, item)
            shutil.copy(file, output_file)
        elif item.split('.')[-1] == 'xml':
            output_file = os.path.join(lables_dir, item)
            shutil.copy(file, output_file)
    

if __name__ == "__main__":
    # data_set = importer.ImportYoloV5(ANOT_DIR, IMG_EXT, NAMES_LIST, IMG_DIR, NAME)
   
    # data_set.export.ExportToVoc(OUT_PATH)
    # yolo_to_pascalvoc("data/baseline_youtube/", "data/baseline_youtube/",(640, 640), NAMES_LIST)

    split_data("data/baseline/xml_labels/", "data/baseline/split_files1", 0.8)
    # yolo_to_pascalvoc("data/baseline/", "data/baseline/",(640, 640), NAMES_LIST)
    # extract_data("data/dummy_label/train", "data/dummy_label/")

    # map_lable_to_class('data/things/things1/labels2', NAMES_LIST)

    # generate_dummy_damples(80, 'data/dumy_images/dummy.png', 'data/dumy_images')