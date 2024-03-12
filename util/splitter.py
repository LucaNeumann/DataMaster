import os
import numpy as np
import shutil


#this program is used to split data into training and validation data, this can happen random or acording to a specific heuristic


def sort_list(list):
    N = len(list)
    value_list = []
    new_list = list.copy()
    for item in list:
        value = int((item.split('_')[1]).split('.')[0])
        value_list.append(value)
    
    indexes = np.argsort(value_list)
    for i in range(N):
        new_list[i] = list[indexes[i]]
    
    return new_list



def split_data(input_dir, output_dir, random_select, split_ratio):
    output_dir_train = os.path.join(output_dir, 'train')
    output_dir_val = os.path.join(output_dir, 'val')

    out_train_img = os.path.join(output_dir_train, 'images')
    out_train_label = os.path.join(output_dir_train, 'labels')

    out_val_img = os.path.join(output_dir_val, 'images')
    out_val_label = os.path.join(output_dir_val, 'labels')

    

    #check if the givven dir is an existing directory

    if os.path.isdir(input_dir):
        #checking the output directory
        if os.path.isdir(output_dir):
            if os.path.isdir(output_dir_train):
                if not os.path.isdir(out_train_img):
                    os.mkdir(out_train_img)
                if not os.path.isdir(out_train_label):
                    os.mkdir(out_train_label)
            else:
                os.mkdir(output_dir_train)
                os.mkdir(out_train_img)
                os.mkdir(out_train_label)
            if os.path.isdir(output_dir_val):
                if not os.path.isdir(out_val_img):
                    os.mkdir(out_val_img)
                if not os.path.isdir(out_val_label):
                    os.mkdir(out_val_label)
            else:
                os.mkdir(output_dir_val)
                os.mkdir(out_val_img)
                os.mkdir(out_val_label)
        else:
            os.mkdir(output_dir)
            os.mkdir(output_dir_train)
            os.mkdir(output_dir_val)
            os.mkdir(out_train_img)
            os.mkdir(out_train_label)
            os.mkdir(out_val_img)
            os.mkdir(out_val_label)

        print("start slecting files")
        img_path = os.path.join(input_dir, 'images')
        label_path = os.path.join(input_dir, 'labels')
        images = sorted(os.listdir(img_path))
        labels = sorted(os.listdir(label_path))
        #print(labels)
        num_of_elements = len(images)
        if sort:
            images = sort_list(images)
            labels = sort_list(labels)
        # select indexes ranging from 0 to num_elements -1
        if random_select:
            index_array = np.random.choice([0, 1], size=(num_of_elements,), p=[1-splitratio, splitratio])
        else:
            num_train_vals = int(splitratio*num_of_elements)
            index_array = np.append(np.ones(num_train_vals), np.zeros(num_of_elements-num_train_vals))
        print(index_array)
        for i in range(num_of_elements):
            current_image = images[i]
            current_label = labels[i]
            if index_array[i] == 1:
                #copy image
                shutil.copy(os.path.join(img_path, current_image), os.path.join(out_train_img, current_image))
                #copy label
                shutil.copy(os.path.join(label_path, current_label), os.path.join(out_train_label, current_label))

            else:
                #copy image
                shutil.copy(os.path.join(img_path, current_image), os.path.join(out_val_img, current_image))
                #copy label
                shutil.copy(os.path.join(label_path, current_label), os.path.join(out_val_label, current_label))
            

    else:
        print("incorrect input file path")



if __name__ == "__main__":
    input_dir = "data/baseline"
    output_dir = "data/baseline"
    random_select = True
    sort = False
    splitratio = 0.8

    split_data(input_dir, output_dir, random_select, splitratio)