'''

Generates bbox's for all the image in a given set of images.

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
HEAD_SF = 80
LIMB_INDEXS = [0, 1, 3, 4, 10, 11, 13, 14]

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

    def __init__(self, point, width, height):
        self.point = point
        self.width = width
        self.height = height


class Limb:

    def __init__(self, top_left_point, bot_right_point, width, height):
        self.top_left_point = top_left_point
        self.bot_right_point = bot_right_point
        self.width = width
        self.height = height


# def create_limb_bbox():

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

    # print(image_metadata)
    # For all the image we have select one and perform some operation
    for current_img_full_path in image_file_path_list:

        counter = counter + 1
        print(counter)

        if 3000 == counter:
            return

        # Remove the full path from the file name
        image_file_name = current_img_full_path.rsplit('/', 1)[-1]

        # Check if the image has metadata
        if image_file_name in image_metadata:

            # # --------------------------------------------------------------------------------------------------
            # # XML
            # # --------------------------------------------------------------------------------------------------
            #
            img = cv2.imread(current_img_full_path)
            img_height, img_width = img.shape[:2]
            #
            #
            #
            root = ET.Element("annotation")
            filename_jpg = ET.SubElement(root, "filename").text = image_file_name
            size = ET.SubElement(root, "size")
            width = ET.SubElement(size, 'width').text = str(img_width)
            height = ET.SubElement(size, 'height').text = str(img_height)
            #
            # # --------------------------------------------------------------------------------------------------
            # # XML
            # # --------------------------------------------------------------------------------------------------

            # Get the meta data for single image
            current_img_metadata = image_metadata[image_file_name]
            # print(current_img_metadata['is_visible'][str(joint_id)])

            for current_pose in current_img_metadata:

                # Now that we are inside the pose, we need to access each joint in that pose.
                for joint in current_pose['joint_pos']:
                    # print(joint)
                    x = current_pose['joint_pos'][joint][0]
                    y = current_pose['joint_pos'][joint][1]
                    # Current joint point
                    p1 = {'x': x, 'y': y}
                    p2 = p1

                    # is_visible = current_pose['is_visible'][joint]
                    # print("Image: ", filename_jpg)
                    # print(current_pose['is_visible'])
                    # if is_visible != 1: continue

                    if int(joint) is not 15:
                        # print("JOINT", joint)
                        x = current_pose['joint_pos'][str(int(joint) + 1)][0]
                        y = current_pose['joint_pos'][str(int(joint) + 1)][1]
                        p2 = {'x': x, 'y': y}

                    # print('Joint ID', joint, 'X:', x, 'Y:', y)

                    angle = math.atan2(p2['y'] - p1['y'], p2['x'] - p1['x']) * 180 / math.pi
                    # cv2.line(img, (int(p1['x']), int(p1['y'])), (int(p2['x']), int(p2['y'])), (255, 255, 255), 2)

                    xc = (p2['x'] + p1['x']) / 2.
                    yc = (p2['y'] + p1['y']) / 2.
                    center = Point((p2['x'] + p1['x']) / 2., (p2['y'] + p1['y']) / 2.)
                    height_joint = math.hypot(p2['x'] - p1['x'], p2['y'] - p1['y'])

                    head = current_pose['head_rect']
                    head_width = head[2] - head[0]
                    limb_width = BBOX_WIDTH * head_width / HEAD_SF

                    if int(joint) in LIMB_INDEXS:
                        print(center.x, center.y, height_joint, limb_width, angle, joint, img)
                        draw_rec_limb_boxes(center.x, center.y, height_joint, limb_width, angle, joint, img)
                        xmin = int(center.x - (height_joint / 2))
                        xmax = int(center.x + (height_joint / 2))
                        ymin = int(center.y - (limb_width / 2))
                        ymax = int(center.y + (limb_width / 2))

                        object = ET.SubElement(root, "object")
                        name = ET.SubElement(object, 'name').text = look_up_table[int(joint)]
                        bndbox = ET.SubElement(object, 'bndbox')
                        xmin = ET.SubElement(bndbox, 'xmin').text = str(xmin)
                        xmax = ET.SubElement(bndbox, 'xmax').text = str(xmax)
                        ymin = ET.SubElement(bndbox, 'ymin').text = str(ymin)
                        ymax = ET.SubElement(bndbox, 'ymax').text = str(ymax)
                        angle = ET.SubElement(bndbox, 'angle').text = str(angle)

                if debug:
                    x = current_pose['joint_pos'][joint][0]
                    y = current_pose['joint_pos'][joint][1]
                    ls = Point(current_pose['joint_pos']['14'][0], current_pose['joint_pos']['14'][1])
                    rs = Point(current_pose['joint_pos']['13'][0], current_pose['joint_pos']['13'][1])
                    lh = Point(current_pose['joint_pos']['3'][0], current_pose['joint_pos']['3'][1])
                    rh = Point(current_pose['joint_pos']['2'][0], current_pose['joint_pos']['2'][1])
                    thorax = Point(current_pose['joint_pos']['7'][0], current_pose['joint_pos']['7'][1])
                    pelvis = Point(current_pose['joint_pos']['6'][0], current_pose['joint_pos']['6'][1])
                    draw_chest(ls, rs, lh, rh, img, image_file_name, thorax, pelvis, OUT_PATH)
                    draw_head(current_pose['head_rect'], img)
                    cv2.imwrite("/home/richard/git/yopo/data/darkflow/ann_images/{}".format(filename_jpg), img)
                    cv2.namedWindow("Display window", cv2.WINDOW_AUTOSIZE)
                    cv2.imshow("Display Window", img)
                    # cv2.waitKey(0)

            filename = filename_jpg.split('.jpg')[0]
            # Write to file
            tree = ET.ElementTree(root)
            XML_OUT = cfg.config['DARKFLOW_XML_OUTPATH']
            tree.write(open('{}{}.xml'.format(XML_OUT, filename), 'w'), encoding='unicode')

        # if debug:
        #     cv2.namedWindow("Display window", cv2.WINDOW_AUTOSIZE)
        #     cv2.imshow("Display Window", img)
        #     cv2.waitKey(0)


# ---------------------------------------------------------------------------------------------------------------------
# Drawing Functions
# ---------------------------------------------------------------------------------------------------------------------


def draw_rec_limb_boxes(x0, y0, width, height, angle, joint_id, img, colour=WHITE):
    if joint_id != 9 and joint_id != 5:
        _angle = angle * math.pi / 180.0
        b = math.cos(_angle) * 0.5
        a = math.sin(_angle) * 0.5
        pt0 = (int(x0 - a * height - b * width), int(y0 + b * height - a * width))
        pt1 = (int(x0 + a * height - b * width), int(y0 - b * height - a * width))
        pt2 = (int(2 * x0 - pt0[0]), int(2 * y0 - pt0[1]))
        pt3 = (int(2 * x0 - pt1[0]), int(2 * y0 - pt1[1]))

        cv2.line(img, pt0, pt1, colour, 3)
        cv2.line(img, pt1, pt2, colour, 3)
        cv2.line(img, pt2, pt3, colour, 3)
        cv2.line(img, pt3, pt0, colour, 3)

        cv2.line(img, pt0, pt0, (255, 0, 0), 10)
        cv2.line(img, pt2, pt2, (255, 0, 0), 10)

        # Show which were are talking about.
        # Draw Text to label the joint.
        cv2.putText(img, org=(int(x0 + TEXT_POSITION_MODIFIER_X),
                              int(y0 + TEXT_POSITION_MODIFIER_Y)),
                    color=colour, text=str(joint_id), fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=2,
                    thickness=3)

        return img


def draw_head(head, img):
    cv2.rectangle(img, (int(head[0]), int(head[1])), (int(head[2]), int(head[3])), (255, 255, 255), 2)


def is_joint_visible(joint_id, pose_meta_data):
    if pose_meta_data[joint_id] == 1:
        return True
    else:
        return False


def create_joint_entry(current_filename, limb_id, x, y, width, height, angle, img, OUT_PATH):
    filename, ext = current_filename.rsplit('.', 1)

    # Convert data into YOLOv2 format
    img_height, img_width = img.shape[:2]
    x = x / img_width
    y = y / img_height
    box_w = width / img_width
    box_h = height / img_height

    output_line = "{} {} {} {} {}\n".format(limb_id, x, y, box_w, box_h)

    with open(OUT_PATH + "{}.txt".format(filename), 'a') as out:
        out.write(output_line)


def draw_chest(right_shoulder, left_shoulder, right_hip, left_hip, img, current_filename, thorax, pelvis, OUT_PATH):
    # Chest Angle
    angle = math.atan2(thorax.y + pelvis.y, thorax.x + pelvis.x) * 180 / math.pi

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
    # h_top = distance_between_points(left_shoulder.x, left_shoulder.y, left_hip.x, left_hip.y)
    # h_bot = distance_between_points(right_shoulder.x, right_shoulder.y, right_hip.x, right_hip.y)
    # height = (h_top + h_bot)
    height = distance_between_points(thorax.x, thorax.y, pelvis.x, pelvis.y)

    draw_rec_limb_boxes(x0=center.x, y0=center.y, width=width, height=height, angle=angle,
                        joint_id=limb_ids['chest'], img=img, colour=RED)

    create_joint_entry(current_filename, limb_ids['chest'], center.x, center.y, width, height, angle, img, OUT_PATH)

    return Chest(center, width, height)


