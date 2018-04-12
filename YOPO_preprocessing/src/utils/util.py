import glob
import shutil
import math
import cv2
import scipy.io as sio
import YOPO_preprocessing.src.main.config as cfg
from darkflow.utils.Point import Point

mat = sio.loadmat(cfg.config['MATLAB_DATA_FILE_PATH'])

# OUTPUT_PATH = cfg.config['OUTPUT_PATH']/
# OUTPUT_PATH =
BOX_W = 50
BOX_H = 50

IMAGES_PATH = "/home/richard/Downloads/images/"
images = glob.glob("/home/richard/Downloads/images/*jpg")
TRAIN_PATH = "/home/richard/git/yopo/data/train_yolo/train_data"
TEST_PATH = "/home/richard/git/yopo/data/train_yolo/test_data/"
YOLO_PATH = "/home/richard/git/yopo/data/train_yolo/"


TEST_IMAGES = glob.glob(
    "/home/richard/git/yopo/data/train_yolo/test_data/*.txt")
TRAIN_IMAGES = glob.glob(
    "/home/richard/git/yopo/data/train_yolo/train_data/labels/*.txt")

'''
A Class that has helpful methods for the data postprocessing.
'''

#  This function is a adapted version of a function from
#  https://github.com/bearpaw/pytorch-pose/blob/master/pose/datasets/mpii.py
def load_matlab_data():
    data = mat.get("results", "")
    """
    Convert annotations mat file to json and save on disk.
    Only persons with annotations of all 16 joints will be written in the json.
    """
    joint_data_fn = 'mpii_human_pose_v1_u12_1.json'
    all_data = {}
    fp = open(joint_data_fn, 'w')

    for i, (anno, train_flag) in enumerate(
            zip(mat['RELEASE']['annolist'][0, 0][0],
                mat['RELEASE']['img_train'][0, 0][0])):

        img_fn = anno['image']['name'][0, 0][0]
        train_flag = int(train_flag)

        if 'annopoints' in str(anno['annorect'].dtype):
            annopoints = anno['annorect']['annopoints'][0]
            head_x1s = anno['annorect']['x1'][0]
            head_y1s = anno['annorect']['y1'][0]
            head_x2s = anno['annorect']['x2'][0]
            head_y2s = anno['annorect']['y2'][0]
            for annopoint, head_x1, head_y1, head_x2, head_y2 in \
                    zip(annopoints, head_x1s, head_y1s, head_x2s, head_y2s):
                if len(annopoint) > 0:
                    head_rect = [float(head_x1[0, 0]),
                                 float(head_y1[0, 0]),
                                 float(head_x2[0, 0]),
                                 float(head_y2[0, 0])]
                    scale = anno['annorect']['scale'][0]
                    obj_pos = anno['annorect']['objpos']
                    # print(obj_pos[0][0])

                    # joint coordinates
                    annopoint = annopoint['point'][0, 0]
                    j_id = [str(j_i[0, 0]) for j_i in annopoint['id'][0]]
                    x = [x[0, 0] for x in annopoint['x'][0]]
                    y = [y[0, 0] for y in annopoint['y'][0]]
                    joint_pos = {}
                    for _j_id, (_x, _y) in zip(j_id, zip(x, y)):
                        joint_pos[str(_j_id)] = [float(_x), float(_y)]
                    # joint_pos = fix_wrong_joints(joint_pos)

                    # visiblity list
                    if 'is_visible' in str(annopoint.dtype):
                        vis = [v[0] if v else [0]
                               for v in annopoint['is_visible'][0]]
                        vis = dict([(k, int(v[0])) if len(v) > 0 else v
                                    for k, v in zip(j_id, vis)])
                    else:
                        vis = None

                    if len(joint_pos) == 16:
                        data_blob = {
                            'filename': img_fn,
                            'train': train_flag,
                            'head_rect': head_rect,
                            'is_visible': vis,
                            'joint_pos': joint_pos,
                            'scale': scale,
                            'obj_pos': obj_pos
                        }

                        if img_fn not in all_data:
                            all_data[img_fn] = []
                        all_data[img_fn].append(data_blob)

                        # all_data[FILE_KEY] = [{}{}{}]
                        # all_
                        # all_data[img_fn] = data_blob

    return all_data


def generate_ground_truth(images, image_data, images_path):
    #  For each given image.
    for x in range(0, len(images)):
        # Get image
        full_image_path = images_path + images[x]
        img = cv2.imread(full_image_path, -1)
        img_width, img_height = img.shape[:2]
        # print("Image Size: ", img_width, img_height)

        current_img_data = image_data[images[x]]

        # For each pose in that given image. {KEY:[{},{}]}
        for pose_key in range(0, len(image_data[images[x]])):
            joint_data = current_img_data[pose_key]['joint_pos']
            # visible_joint = current_img_data[pose_key]['is_visible']
            # head_data = current_img_data[pose_key]['head_rect']

            print("------------------------------")
            #  For each joint in that given pose.
            # Each class needs it own line the truth file.
            for joint_id in range(0, 16):
                # print(joint_data[str(joint_id)])

                # Check if joint is viable in the image.
                if current_img_data['is_visible'] == 1:
                    # YOLO Data
                    yolo_class = joint_id
                    yolo_x = joint_id[0] / img_width
                    yolo_y = joint_id[1] / img_height
                    yolo_w = BOX_W / img_width
                    yolo_h = BOX_H / img_height

                # print(convert(), joint_id)
            print("------------------------------")


# todo add these to a config file. training speed dif


def prepare_train_and_test_data():
    for x in TRAIN_IMAGES:
        train_img_path = x.split('.')[0] + ".jpg"
        # break to next line
        train_img_path = train_img_path + "\n"

        filename = x.rsplit('/', 1)[-1].split('.')[0] + ".jpg"
        # print("{}{}".format(IMAGES_PATH, filename, TRAIN_PATH))
        shutil.copy2("{}{}".format(
            cfg.config['IMAGE_PATH'], filename), "{}".format(TRAIN_PATH))

        with open(OUTPUT_PATH + "{}".format("train.txt"), 'a') as out:
            out.write(train_img_path)

    for y in TEST_IMAGES:
        train_img_path = y.split('.')[0] + ".jpg"
        # break to next line
        train_img_path = train_img_path + "\n"

        filename = y.rsplit('/', 1)[-1].split('.')[0] + ".jpg"

        shutil.copy2("{}{}".format(
            cfg.config['IMAGE_PATH'], filename), "{}".format(TEST_PATH))
        # print("Test File out path {}". format(OUTPUT_PATH + "{}".format("test.txt")))
        with open(OUTPUT_PATH + "{}".format("test.txt"), 'a') as out:
            out.write(train_img_path)


def darkflow_sort_images():
    TRAIN_IMAGES = glob.glob(
        "/home/richard/git/yopo/data/darkflow/labels/*xml")
    for x in TRAIN_IMAGES:
        train_img_path = x.split('.')[0] + ".jpg"
        # break to next line
        train_img_path = train_img_path + "\n"

        filename = x.rsplit('/', 1)[-1].split('.')[0] + ".jpg"
        # print("{}{}".format(IMAGES_PATH, filename, TRAIN_PATH))
        shutil.copy2("{}{}".format(cfg.config['IMAGE_PATH'], filename), "{}".format(
            cfg.config['DARKFLOW_IMAGES_OUTPATH']))


# Get point of a box when given a centre point, width, height and angle.

def draw_rec_centre_point(x0, y0, width, height, angle, img, colour=(255, 255, 255)):
    _angle = angle * math.pi / 180.0
    # _angle = angle
    b = math.cos(_angle) * 0.5
    a = math.sin(_angle) * 0.5
    pt0 = (int(x0 - a * height - b * width), int(y0 + b * height - a * width))
    pt1 = (int(x0 + a * height - b * width), int(y0 - b * height - a * width))
    pt2 = (int(2 * x0 - pt0[0]), int(2 * y0 - pt0[1]))
    pt3 = (int(2 * x0 - pt1[0]), int(2 * y0 - pt1[1]))

    cv2.line(img, pt0, pt1, colour, 2)
    cv2.line(img, pt1, pt2, colour, 2)
    cv2.line(img, pt2, pt3, colour, 2)
    cv2.line(img, pt3, pt0, colour, 2)

    cv2.line(img, pt0, pt0, (255, 255, 255), 4)
    cv2.line(img, pt2, pt2, (255, 255, 255), 4)
    return img


def draw_polygon(image, pts, colour=(51, 255, 51), thickness=2):
    """
    Draws a rectangle on a given image.

    :param image: What to draw the rectangle on
    :param pts: Array of point objects
    :param colour: Colour of the rectangle edges
    :param thickness: Thickness of the rectangle edges
    :return: Image with a rectangle
    """

    for i in range(0, len(pts)):
        n = (i + 1) if (i + 1) < len(pts) else 0
        cv2.line(image, (pts[i].x, pts[i].y), (pts[n].x, pts[n].y), colour, thickness)

    return image


def distance_between_points(x1, y1, x2, y2):
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist


def midpoint(x1, y1, x2, y2):
    x, y = (x1 + x2) / 2, (y1 + y2) / 2
    return Point(x, y)