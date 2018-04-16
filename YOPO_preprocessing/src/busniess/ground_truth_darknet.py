import cv2

WIDTH_INDEX = 0
HEIGHT_INDEX = 1
IMAGE_OUTPUT_PATH = "../../output/box_images/"


def generate_yolo_data_files(img, img_data, image_index, enable_img_output, is_train):
    if (is_train):
        TEXT_OUTPUT = "../../data/train_yolo/train_data/labels/"
    else:
        TEXT_OUTPUT = "../../data/train_yolo/test_data/images/"

    # Take centre point and a width and height
    image_truth = []
    # img - image file load from openCV
    # img_data - large dict contain all the meta for every image, img_data['img'] ->
    # returns you a array of object for each pose in the image.

    # An array of image meta data for the img
    # print(image_index.rsplit('/', 1)[-1])
    image_file_name = image_index.rsplit('/', 1)[-1]
    # print(image_file_name)
    image_id, extension = image_file_name.split('.')
    # print("image Id: {}".format(image_id))

    img_height, img_width = img.shape[:2]
    # print("Height {}, Width {}".format(img_height,img_width))
    # print("IMAGE DATA: {}".format(img_data))

    if image_file_name in img_data:

        image_set = img_data[image_file_name]
        # print(image_set[])

        for current_pose in image_set:
            # todo add head data to output image.
            # Draw head bounding box - .x1, .y1, .x2, .y2
            # cv2.rectangle(img, (int(head_data[0]), int(head_data[1])), (int(head_data[2]), int(head_data[3])),
            #               HEAD_REC_COLOUR)

            for joint_id in range(0, 16):

                if current_pose['is_visible'][str(joint_id)] == 1:
                    x = current_pose['joint_pos'][str(joint_id)][WIDTH_INDEX]
                    y = current_pose['joint_pos'][str(joint_id)][HEIGHT_INDEX]
                    width = 50
                    height = 50

                    top_left = int((x - width / 2)), int((y - height / 2))
                    bottom_right = int((x + width / 2)), int((y + height / 2))

                    # Plot a bounding box around a joint(s)

                    if enable_img_output:
                        cv2.rectangle(img, top_left, bottom_right, (255, 255, 255), 2)

                    # write to text file for ground truth data
                    # joint_id |  x | y | w | h

                    # Convert to YOLO format.
                    # todo need to calculate the best box size currently just guessing.
                    # todo also need to make "50" in config file or constant.

                    x = x / img_width
                    y = y / img_height
                    box_w = 50 / img_width
                    box_h = 50 / img_height
                    # todo adding +1 testing a yolo config please remove when done.
                    truth_line = '{} {} {} {} {} \n'.format(joint_id + 1, x, y, box_w, box_h)
                    with open(TEXT_OUTPUT + "{}.txt".format(image_id), 'a') as out:
                        out.write(truth_line)
            if enable_img_output:
                cv2.imwrite(IMAGE_OUTPUT_PATH + image_file_name, img)


# todo set box w,h based of the distances between joints in the images to account for each of the differences in sizes
# todo in each of the images.
def get_yolo_coordinates(x, y, box_w, box_h, img):
    img_height, img_width = img.shape[:2]

    x = x / img_width
    y = y / img_height
    box_w = 50 / img_width
    box_h = 50 / img_height

    return x, y, box_w, box_h


def create_ground_truth(images, data, enable_img_output=True, limit=500):
    # Training set
    # todo change to config number
    for x in range(0, 100):
        # print(IMAGES_PATH + x)
        print(images[x], x)
        img = cv2.imread(images[x], -1)
        # print(x)
        generate_yolo_data_files(img, data, images[x], enable_img_output, is_train=True)



    #
    # # Test Set
    # todo change to config number
    # for y in range(101, 121):
    #     # print(IMAGES_PATH + x)
    #     img = cv2.imread(images[y], -1)
    #     generate_yolo_data_files(img, data, images[y], enable_img_output, is_train=False)
