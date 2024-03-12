import os
import youtube_dl
import cv2
import detectors
import numpy as np
import matplotlib.pyplot as plt
import detectron2.structures.boxes as bbox
import torch
import random

from pynput.keyboard import Key, Listener



"""
This file contains the functions needed to extract labeled data from images.
Important note: make sure that the predictor used returns the class and box predictions of an image as two lists containing the 
elements."""



def get_index(list, item):
    #returns the index of the item in the list
    for i, x in enumerate(list):
        if item == x:
            return i
    return -1

class key_board():
    def __init__(self, keys=['n', 'y', '1', '2', '3', '4', '5']):
        self.selection = 0
        self.keys = keys

    def on_press(self,key):
        pass
    def on_release(self,key):
        try:
            key_char = key.char
            index = get_index(self.keys, key_char)
            print("index", index)
            if index != -1:
                self.selection = index
                return False
            else:
                pass
        except:
            pass
        
    def start_keyboard(self):
        with Listener(
            on_press=self.on_press,
            on_release=self.on_release) as listener:
            listener.join()
        print("Selef.selection", self.selection)
        return self.selection


def download_video(vid, output_dir):
        # Use youtube_dl to download the video
    ydl_opts = {'quiet':True, 'ignoreerrors':True, 'no_warnings':True,
                'format': 'best[ext=mp4]',
                'outtmpl':output_dir+'/'+vid+'_temp.mp4'}
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download(['youtu.be/'+vid])
    video_path = video_path = output_dir+'/'+vid+'_temp.mp4'
    return video_path


def splice_video(video_path, save_dir, sample_rate):

    if not os.path.isdir(save_dir):
        print("making temporal output_dir")
        os.mkdir(save_dir)

    if os.path.exists(video_path):
        # Use opencv to open the video
        capture = cv2.VideoCapture(video_path)
        fps, total_f = capture.get(5), capture.get(7)

        # Get time stamps (in seconds) for every frame in the video
        # This is necessary because some video from YouTube come at 29.99 fps,
        # other at 30fps, other at 24fps
        timestamps = [i/float(fps) for i in range(int(total_f))]
        fs_devider = max(int(fps/sample_rate), 1)
        #list of the indexes is also sufficccient
        index_list = [i*fs_devider for i in range(int(len(timestamps)/fs_devider))]
        capture.set(cv2.CAP_PROP_FRAME_COUNT, total_f)
        all_images = []
        for i, index in enumerate(index_list):
            # Get the actual image corresponding to the frame
            capture.set(cv2.CAP_PROP_POS_FRAMES,index)
            ret, image = capture.read()
            image_name = "image" + str(i) + ".jpg"
            try:
                cv2.imwrite(os.path.join(save_dir, image_name), image)
            except:
                print("image capture was empty")
        capture.release()
        return all_images

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[0] = (x[0] + x[2]) / 2  # x center
    y[1] = (x[1] + x[3]) / 2  # y center
    y[2] = x[2] - x[0]  # width
    y[3] = x[3] - x[1]  # height
    return y


def write_annotation(prediction, label_dir, prefix, index, image):
    """
    Small function to write down labels in the yolo format.
    Input:
        -prediction: tuple(list(classes), list(boxes))
        -label_dir: str(output dir for the labels)
        -prefix: str(placed before the image and label to give somme context)
        -index: int(the number of the label in the dataset)
        -image: array_like"""
    lines = []
    width = np.shape(image)[1]
    height = np.shape(image)[0]
    classes, boxes = prediction
    print(width, height)
    for i,box in enumerate(boxes):
        box = xyxy2xywh(box)
        label = classes[i]
        line = str(int(label)) + ' ' + str(float(box[0])/width) + ' ' + str(float(box[1])/height) + ' ' + str(float(box[2])/width) + ' ' + str(float(box[3])/height) + '\n'
        lines.append(line)
    f = open(label_dir + prefix + '_' + str(index) + '.txt', 'w')
    f.writelines(lines)
    f.close()


def plot_one_box(x, img, color=None, label=None, line_thickness=3): 
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def eddit_prediction(prediction, image, labels):
    """
    This function is used to eddit labels. first the complete labeled image is dirplayed.
    The user can then select the label or reject it entirly. 
    If the labeled image is selected is proceeds to the next process where each object is individualy added to the label.
    Here the label can be adjusted, accepted or rejected depending on the users choise. 
    
    Input: 
        -prediction: tuple(list(classes), list(boxes))
        -image: array_like image
        -labels: the custom labels we want to use to label data
    Output:
        -result: tuple(list(classes), list(boxes))"""
    classes, boxes = prediction
    #initialize the result arrays for storing the desired boxes and labels
    result_boxes = []
    result_classes = []
    #in this part we display different labels and pass or fail them
    num_pred = len(boxes)
    img = image.copy()
    key = key_board()
    select = 0
    for i in range(num_pred):
        img_temp = img.copy()
        print("label", classes)
        plot_one_box(boxes[i], img_temp, color=None, label=None)
        disp_image = img_temp.copy()
        disp_image = cv2.resize(disp_image, (900,900))
        cv2.imshow("image", disp_image)
        cv2.waitKey(500)
        select = key.start_keyboard()
        cv2.destroyAllWindows()
        #inser decision making system to ensure that we can select
        if select == 0:
            pass
        elif select == 1:
            result_boxes.append(boxes[i])
            result_classes.append(classes[i])
            img = img_temp
        else:
            result_boxes.append(boxes[i])
            result_classes.append(labels[select - 2])
            img = img_temp

    return result_classes, result_boxes



def create_labels_manual(detector, input_dir, output_dir, prefix, custom_labels):
    """
    Function used to generate labels. This is done using a machine learning model and human intervention to check and adjust the labels.
    Input:
        -detector: this is a class that contains a predict method and a visualize method for images and predictions respectively
        -input_dir: str(the directory where the rw images are saved)
        -output_dir: str(this is the directory to where the usefull labels and images will be written)
        -prefix: str(to give somme context to the data)
        -custom_labels: list(these labels are mapped to the keys)
    Output:
        -the outputs are written to file the function itself returns nothing"""
    
    image_directory = output_dir + '/images/'
    label_directory = output_dir + '/labels/'

    if not os.path.isdir(input_dir):
        print("input directory is a wrong directory")
        exit()

    if os.path.isdir(output_dir + '/'):
        pass
    else:
        os.mkdir(output_dir + '/')

    if os.path.isdir(image_directory):
        pass
    else:
        os.mkdir(image_directory)

    if os.path.isdir(label_directory):
        pass
    else:
        os.mkdir(label_directory)


    files = os.listdir(image_directory)
    index = 0
    for file in files:
        file_idx = int(file.split('_')[-1].split('.')[0])
        index = max(index, file_idx+1)

    key = key_board()
    #make prediction
    images = os.listdir(input_dir)

    for image in images:

        image = cv2.imread(os.path.join(input_dir, image))
        prediction = detector.predict(image)

        img = detector.visualize(image, prediction)
        width = np.shape(img)[1]
        height = np.shape(img)[0]
        print(width, height)
        img = cv2.resize(img, (900, 900))
        cv2.imshow("image", img)
        cv2.waitKey(500)
        selection = key.start_keyboard()

        cv2.destroyAllWindows()

        if selection == 1:
            classes, boxes = eddit_prediction(prediction, image, custom_labels)
            image_dir = image_directory + prefix + '_' + str(index) + '.jpg'
            cv2.imwrite(image_dir, image)
            write_annotation((classes, boxes), label_directory, prefix, index, image)
            index += 1
        else:
            pass



def create_labels_automatic(detector, input_dir, output_dir, prefix, class_list):
    """
    Function used to generate labels. This is done using a object detection moddel the labels are not checke by a human and
    therfore the quality will be lower but we can generate more data.
    Input:
        -detector: this is a class that contains a predict method and a visualize method for images and predictions respectively
        -input_dir: str(the directory where the rw images are saved)
        -output_dir: str(this is the directory to where the usefull labels and images will be written)
        -prefix: str(to give somme context to the data)
        -class_list: list(this list contains the classes that we want to detect in the images.
                            This way we can ignore somme background objects)
    Output:
        -the outputs are written to file the function itself returns nothing"""
    
    
    image_directory = output_dir + '/images/'
    label_directory = output_dir + '/labels/'
    if not os.path.isdir(input_dir):
        print("wrong input directory")
        exit()
    else:
        images = os.listdir(input_dir)

    if os.path.isdir(output_dir + '/'):
        pass
    else:
        os.mkdir(output_dir + '/')

    if os.path.isdir(image_directory):
        pass
    else:
        os.mkdir(image_directory)

    if os.path.isdir(label_directory):
        pass
    else:
        os.mkdir(label_directory)


    files = os.listdir(image_directory)
    index = 0
    for file in files:
        file_idx = int(file.split('_')[-1].split('.')[0])
        index = max(index, file_idx+1)


    #load the model
    selection = 1
    #make prediction
    for image in images:
        image = cv2.imread(os.path.join(input_dir, image))
        prediction = detector.predict(image)
        print(prediction["instances"].pred_classes)
        print(prediction["instances"].pred_boxes)

        print("selection", selection)

        classes, boxes = extract_prediction(prediction, image, class_list)
        #only write the label if there is a detection present
        print(classes)
        print("len of classes", len(classes))
        if len(classes) > 0:
            image_dir = image_directory + prefix + '_' + str(index) + '.jpg'
            cv2.imwrite(image_dir, image)
            write_annotation((classes, boxes), label_directory, prefix, index, image)
            index += 1




def extract_prediction(prediction, image, class_list):
    """
    Function used to select ceratin predictions and leave the rest out
    Inputs:
        -predcition: tupple(list(classes), list(boxes))
        -image: the raw image used to make the prediction
        -class_list: list(contains the classes that we want to detect)
        
    Output:
        -result: tupple(list(classes), list(boxes))"""
    
    
    #extract the prediction results and store them in a list
    classes, boxes = prediction
    #initialize the result arrays for storing the desired boxes and labels
    result_boxes = []
    result_classes = []
    #in this part we display different labels and pass or fail them
    num_pred = len(boxes)
    img = image.copy()
    #key = key_board()
    select = 0
    for i in range(num_pred):
        if classes[i] in class_list:
            result_boxes.append(boxes[i])
            result_classes.append(classes[i])

    return result_classes, result_boxes