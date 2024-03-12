import util.generator as label
import os
import shutil
"""
This python script is used to create labeled data from videos. The labels can be created in a supervised way, 
this means that a person has to confirm the generation of each individual label for an image.
The other option is to generate labels and automaticaly reject empty frames. In this case we assme that the model used for
generating the predictions is sufficiently accurate for our purpose.
"""

VIDEO_PATH = 'data/baseline/raw_data/bike.mp4'
TEMP_IMAGES = 'data/baseline/temp_images'
OUTPUT_PATH = 'data/baseline/bike'
PREFIX = 'bike'
#CUSTOM_LABELS = [1,2,3,4,5,6,7,8]
SELECTED_LABELS = [1]
SAMPLE_RATE = 10



def multiple_videos(video_dir, temp_dir, output_dir, prefix, select_classes):
    for path in os.listdir(video_dir):
        total_path = os.path.join(video_dir, path)
        label.splice_video(total_path, temp_dir, sample_rate=SAMPLE_RATE)
        label.create_labels_no_correction(temp_dir, output_dir, prefix, select_classes)

def single_video(video_dir, temp_dir, output_dir, prefix, select_classes):
    label.splice_video(video_dir, temp_dir, sample_rate=SAMPLE_RATE)
    label.create_labels_no_correction(temp_dir, output_dir, prefix, select_classes)


def create_labels(video_dir, temp_dir, output_dir, prefix, select_classes):
    video_dir = os.path.join(os.getcwd(), video_dir)
    temp_dir = os.path.join(os.getcwd(), temp_dir)
    output_dir = os.path.join(os.getcwd(), output_dir)
    if os.path.isdir(video_dir):
        multiple_videos(video_dir, temp_dir, output_dir, prefix, select_classes)
    if os.path.isfile(video_dir):
        single_video(video_dir, temp_dir, output_dir, prefix, select_classes)
    
    shutil.rmtree(temp_dir)


if __name__ == "__main__":
    create_labels(VIDEO_PATH, TEMP_IMAGES, OUTPUT_PATH, PREFIX, SELECTED_LABELS)