'''

MATLAB_DATA_FILE_PATH - THE PATH WHERE THE MATLAB FILE IS THAT CONTAINS THE GROUND TRUTH DATA ABOUT THE IMAGE SET.
IMAGE_PATH - The path of the images training and testing sets.
TRAINING_OUT_PATH - The path of the output files for the training text files and images.
TESTING_OUT_PATH - The path of the output files for the training text files and images.
TRAIN_SET_IMAGES_NUM - The amount of images you want to train on.
TEST_SET_IMAGES_NUM - The amount of images you want to validate the network on.


'''

config = {
    'MATLAB_DATA_FILE_PATH': "../../data/mpii_human_pose_v1_u12_1.mat",
    # 'MATLAB_DATA_FILE_PATH': "/home/richard/git/yopo/data/mpii_human_pose_v1_u12_1.mat",
    'TRAIN_SET_IMAGES_NUM': 5,
    'TEST_SET_IMAGES_NUM': 1,
    'IMAGE_PATH': "/media/richard/Data/images/",
    'TRAINING_OUTPUT_PATH': "../../data/train_yolo/train_data/labels/",
    'TESTING_OUT_PATH': "../../data/train_yolo/test_data/",
    'YOLO_PATH': "../../data/train_yolo/",
    'CV_CONFIG': {
        'TEXT_POSITION_MODIFIER_X': -5,
        'TEXT_POSITION_MODIFIER_Y': 20,
        # # BGR colour
        'POINT_THICKNESS': 2,
        'TEXT_THICKNESS': 2,
        'TEXT_SCALE': 1.25,
        'TEXT_COLOUR': (255, 0, 255),
        'HEAD_REC_COLOUR': (255, 255, 0),
    },
    'OUTPUT_PATH': "../../output/",
    'DARKFLOW_XML_OUTPATH': "../../data/darkflow/labels/",
    'DARKFLOW_IMAGES_OUTPATH': "../../data/darkflow/images/"

}
