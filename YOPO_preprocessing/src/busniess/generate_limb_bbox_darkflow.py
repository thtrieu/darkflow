'''

Generates bbox's for all the image in a given set of images.

MPII Dataset Only!

'''

import math
import cv2
import YOPO_preprocessing.src.main.config as cfg
import xml.etree.cElementTree as ET

# COLOUR SPACE BGR
from YOPO_preprocessing.src.busniess.Point import Point
from YOPO_preprocessing.src.utils.util import midpoint, distance_between_points

TRAIN_SET_SIZE = 500
TEST_SET_SIZE = 100
TEXT_POSITION_MODIFIER_X = 50
TEXT_POSITION_MODIFIER_Y = 50
WHITE = 255, 255, 255
RED = 0, 0, 255
BBOX_WIDTH = 50
HEAD_SF = 70
LIMB_INDEXS = [0, 1, 3, 4, 10, 11, 13, 14]
IMAGE_PROCESS_LIMIT = 4000
HALF = 2
FLOAT_HALF = 2.0

object_id = {0: 'left-lower-leg',
             1: 'left-upper-leg',
             2: 'right-lower-leg',
             3: 'right-upper-leg',
             4: 'left-lower-arm',
             5: 'left-upper-arm',
             6: 'right-lower-arm',
             7: 'right-upper-arm',
             8: 'chest',
             9: 'head'
             }

limb_ids = {
    'left-lower-leg': 0,
    'left-upper-leg': 1,
    'right-lower-leg': 2,
    'right-upper-leg': 3,
    'left-lower-arm': 4,
    'left-upper-arm': 5,
    'right-lower-arm': 6,
    'right-upper-arm': 7,
    'chest': 8,
    'head': 9
}

look_up_table = {0: 'left-lower-leg',
                 1: 'left-upper-leg',
                 3: 'right-lower-leg',
                 4: 'right-upper-leg',
                 10: 'left-lower-arm',
                 11: 'left-upper-arm',
                 13: 'right-lower-arm',
                 14: 'right-upper-arm'
                 }


class Chest:

    def __init__(self, centre, width, height, angle):
        self.centre = centre
        self.width = width
        self.height = height
        self.angle = angle


class Limb:

    def __init__(self, top_left_point, bot_right_point, width, height):
        self.top_left_point = top_left_point
        self.bot_right_point = bot_right_point
        self.width = width
        self.height = height


#  image_file_path_list - A list of all the image with the fill path names.
#  image_metadata - a python dictionary that contains all pose data for a given image.
def generate_limb_data(image_file_path_list, image_metadata, train=True, debug=False):
    counter = 0

    if train:
        limit = cfg.config['TRAIN_SET_IMAGES_NUM']
        OUT_PATH = cfg.config['TRAINING_OUTPUT_PATH']
        image_file_path_list = image_file_path_list[cfg.config['TEST_SET_IMAGES_NUM']:]
    else:
        limit = cfg.config['TEST_SET_IMAGES_NUM']
        OUT_PATH = cfg.config['TESTING_OUT_PATH']
        image_file_path_list = image_file_path_list[:cfg.config['TEST_SET_IMAGES_NUM']]

    # For all the image we have select one and perform some operation
    for current_img_full_path in image_file_path_list:

        counter = counter + 1

        # Limits the amount of data
        if counter > 6000:
            return

        # Remove the full path from the file name
        image_file_name = current_img_full_path.rsplit('/', 1)[-1]

        # Check if the image has metadata
        if image_file_name in image_metadata:

            # get image
            img = cv2.imread(current_img_full_path)
            img_height, img_width = img.shape[:2]

            root = ET.Element("annotation")
            filename_jpg = ET.SubElement(root, "filename").text = image_file_name
            size = ET.SubElement(root, "size")
            width = ET.SubElement(size, 'width').text = str(img_width)
            height = ET.SubElement(size, 'height').text = str(img_height)

            # Get the meta data for single image
            current_img_metadata = image_metadata[image_file_name]

            for current_pose in current_img_metadata:

                # Edge case: chest made up of many points
                chest = create_chest(current_pose)

                # Get min and max point for chest
                xmin_chest = int(chest.centre.x - (chest.height / HALF))
                xmax_chest = int(chest.centre.x + (chest.height / HALF))
                ymin_chest = int(chest.centre.y - (chest.width / HALF))
                ymax_chest = int(chest.centre.y + (chest.width / HALF))
                create_XML_entry(root, xmin_chest, xmax_chest, ymin_chest, ymax_chest, chest.angle, name="chest")

                # Head
                head = current_pose['head_rect']
                head_angle = math.degrees(math.atan2(head[3] - head[1], head[2] - head[0]))
                create_XML_entry(root, head[0], head[2], head[1], head[3], head_angle, name="head")

                # Arms and legs
                for joint in current_pose['joint_pos']:

                    # Get Joint meta data
                    x = current_pose['joint_pos'][joint][0]
                    y = current_pose['joint_pos'][joint][1]
                    # Current joint point
                    p1 = Point(x, y)
                    p2 = p1

                    # Skip joint if it's not visible.
                    if current_pose['is_visible'][joint] == 0:
                        continue

                    if int(joint) is not 15:
                        x = current_pose['joint_pos'][str(int(joint) + 1)][0]
                        y = current_pose['joint_pos'][str(int(joint) + 1)][1]
                        p2 = Point(x, y)

                    # Find Centre points and height of the joint
                    center = Point((p2.x + p1.x) / FLOAT_HALF, (p2.y + p1.y) / FLOAT_HALF)
                    height_joint = math.hypot(p2.x - p1.x, p2.y - p1.y)

                    # Width
                    head_width = head[2] - head[0]
                    limb_width = BBOX_WIDTH * head_width / HEAD_SF

                    # Get Angle
                    angle = math.degrees(math.atan2(p2.y - p1.y, p2.x - p1.x))

                    if int(joint) in LIMB_INDEXS:
                        if int(joint) != 9 and int(joint) != 5:
                            # Draw Head, Arms, Legs, Chest
                            draw_rec_limb_boxes(center.x, center.y, height_joint, limb_width, angle, img)
                            draw_head(head, img)
                            draw_rec_limb_boxes(chest.centre.x, chest.centre.y, chest.height, chest.width,
                                                chest.angle, img)

                        xmin = int(center.x - (height_joint / HALF))
                        xmax = int(center.x + (height_joint / HALF))
                        ymin = int(center.y - (limb_width / HALF))
                        ymax = int(center.y + (limb_width / HALF))

                        name = look_up_table[int(joint)]
                        create_XML_entry(root, xmin, xmax, ymin, ymax, angle, name)

                if debug:
                    cv2.imwrite("../../data/ann_images/{}".format(filename_jpg), img)
                    cv2.namedWindow("Display window", cv2.WINDOW_AUTOSIZE)
                    cv2.imshow("Display Window", img)
                    cv2.waitKey(0)

            filename = filename_jpg.split('.jpg')[0]
            # Write to file
            tree = ET.ElementTree(root)
            XML_OUT = cfg.config['DARKFLOW_XML_OUTPATH']
            tree.write(open('{}{}.xml'.format(XML_OUT, filename), 'w'), encoding='unicode')


# ---------------------------------------------------------------------------------------------------------------------
# Drawing Functions
# ---------------------------------------------------------------------------------------------------------------------


def draw_rec_limb_boxes(x0, y0, width, height, angle, img, colour=WHITE, thickness=2):

    _angle = math.radians(angle)
    b = math.cos(_angle) * 0.5
    a = math.sin(_angle) * 0.5
    pt0 = (int(x0 - a * height - b * width), int(y0 + b * height - a * width))
    pt1 = (int(x0 + a * height - b * width), int(y0 - b * height - a * width))
    pt2 = (int(2 * x0 - pt0[0]), int(2 * y0 - pt0[1]))
    pt3 = (int(2 * x0 - pt1[0]), int(2 * y0 - pt1[1]))

    cv2.line(img, pt0, pt1, colour, thickness)
    cv2.line(img, pt1, pt2, colour, thickness)
    cv2.line(img, pt2, pt3, colour, thickness)
    cv2.line(img, pt3, pt0, colour, thickness)

    cv2.line(img, pt0, pt0, (255, 0, 0), thickness)
    cv2.line(img, pt2, pt2, (255, 0, 0), thickness)

    return img


def draw_head(head, img):
    cv2.rectangle(img, (int(head[0]), int(head[1])), (int(head[2]), int(head[3])), (255, 255, 255), 2)


def is_joint_visible(joint_id, pose_meta_data):
    if pose_meta_data[joint_id] == 1:
        return True
    else:
        return False


def create_joint_entry(current_filename, limb_id, x, y, width, height, angle, img, OUT_PATH, limb_width):
    filename, ext = current_filename.rsplit('.', 1)

    # Convert data into YOLOv1 format
    img_height, img_width = img.shape[:2]
    x = x / img_width
    y = y / img_height
    box_w = width / img_width
    box_h = height / img_height

    output_line = "{} {} {} {} {}\n".format(limb_id, x, y, box_w, box_h)

    with open(OUT_PATH + "{}.txt".format(filename), 'a') as out:
        out.write(output_line)


def _create_chest(right_shoulder, left_shoulder, right_hip, left_hip, thorax, pelvis):
    # Chest Angle
    angle = math.degrees(math.atan2(thorax.y - pelvis.y, thorax.x - pelvis.x))

    # Calculates the centre of the chest
    mid_left_y = midpoint(left_shoulder.x, left_shoulder.y, left_hip.x, left_hip.y)
    mid_right_y = midpoint(right_shoulder.x, right_shoulder.y, right_hip.x, right_hip.y)

    # Mean Distance between points
    # center = Point(((mid_left_y[0] + mid_right_y[0]) / 2), (mid_right_y[1] + mid_left_y[1]) / 2)
    center = midpoint(thorax.x, thorax.y, pelvis.x, pelvis.y)

    # Width
    w_top = distance_between_points(left_shoulder.x, left_shoulder.y, right_shoulder.x, right_shoulder.y)
    w_bot = distance_between_points(left_hip.x, left_hip.y, right_hip.x, right_hip.y)
    width = (w_top + w_bot)

    # Height
    height = distance_between_points(thorax.x, thorax.y, pelvis.x, pelvis.y)

    return Chest(center, width, height, angle)


def create_chest(current_pose):
    ls = Point(current_pose['joint_pos']['14'][0], current_pose['joint_pos']['14'][1])
    rs = Point(current_pose['joint_pos']['13'][0], current_pose['joint_pos']['13'][1])
    lh = Point(current_pose['joint_pos']['3'][0], current_pose['joint_pos']['3'][1])
    rh = Point(current_pose['joint_pos']['2'][0], current_pose['joint_pos']['2'][1])
    thorax = Point(current_pose['joint_pos']['7'][0], current_pose['joint_pos']['7'][1])
    pelvis = Point(current_pose['joint_pos']['6'][0], current_pose['joint_pos']['6'][1])

    return _create_chest(ls, rs, lh, rh, thorax, pelvis)


def create_XML_entry(root, xmin, xmax, ymin, ymax, angle, name):
    object = ET.SubElement(root, "object")
    name = ET.SubElement(object, 'name').text = name
    bndbox = ET.SubElement(object, 'bndbox')
    xmin = ET.SubElement(bndbox, 'xmin').text = str(xmin)
    xmax = ET.SubElement(bndbox, 'xmax').text = str(xmax)
    ymin = ET.SubElement(bndbox, 'ymin').text = str(ymin)
    ymax = ET.SubElement(bndbox, 'ymax').text = str(ymax)
    angle = ET.SubElement(bndbox, 'angle').text = str(angle)
