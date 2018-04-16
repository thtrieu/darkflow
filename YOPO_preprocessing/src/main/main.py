
import glob
import time
from YOPO_preprocessing.src.main.config import config
from YOPO_preprocessing.src.busniess.generate_limb_bbox_darkflow import generate_limb_data
from YOPO_preprocessing.src.busniess.ground_truth_darknet import create_ground_truth
from YOPO_preprocessing.src.utils.util import load_matlab_data, darkflow_sort_images, prepare_train_and_test_data

images = glob.glob("{}*jpg".format(config['IMAGE_PATH']))

'''
Process chain 

1. Take the MatLab data files and convert them to python dict 
2. Take image set and generate ground truths in the YOLO or Darkflow format 
3. Output bounding boxes using YOLO ground truth information.
4. Move Images and text files into folder ready for the network.

'''

# Darkflow is the python version which requires different pre-processing then darknet the C CUDA version.
darkflow = True

if __name__ == "__main__":

    if len(images) == 0:
        print("ERROR: CANNOT FIND IMAGE DATA!")
        exit(-1)

    # Load image meta data.
    data = load_matlab_data()
    start_time = time.time()
    if darkflow:
        generate_limb_data(image_file_path_list=images, image_metadata=data, train=True)
        darkflow_sort_images()

    else:
        # Generate the ground_truth_text_files.
        create_ground_truth(images, data, limit=500)

        # Move images that have just had ground truth text files created for them into the same folder ../../train_yolo/
        # todo - The program need to ran twice for this to work. YOPO_preprocessing-7
        prepare_train_and_test_data()

    print("Finished in %s seconds " % int(time.time() - start_time))


