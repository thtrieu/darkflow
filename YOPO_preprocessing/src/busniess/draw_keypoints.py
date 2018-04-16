import cv2

IMAGE_OUTPUT_PATH = "../../output/images/"

TEXT_POSITION_MODIFIER_X = -5
TEXT_POSITION_MODIFIER_Y = 20
# BGR colour
POINT_THICKNESS = 2
TEXT_THICKNESS = 2
TEXT_SCALE = 1.25
TEXT_COLOUR = 255, 0, 255
HEAD_REC_COLOUR = 255, 255, 0


# if joint is viable it's red otherwise it's green
def joint_colour(num):
    if num == 1:
        return 0, 0, 255
    else:
        return 0, 255, 0


def draw_data_point_on_images(images, image_path, img_data):
  
    for x in range(0, len(images)):
        # Get image
        full_image_path = image_path + images[x]
        img = cv2.imread(full_image_path, -1)

        image_file_name = images[x].rsplit('/', 1)[-1]

        if image_file_name in img_data:

            current_img_data = img_data[image_file_name]

            for pose_key in range(0, len(img_data[image_file_name])):
                joint_data = current_img_data[pose_key]['joint_pos']
                visible_joint = current_img_data[pose_key]['is_visible']
                head_data = current_img_data[pose_key]['head_rect']

                # Draw head bounding box - .x1, .y1, .x2, .y2
                cv2.rectangle(img, (int(head_data[0]), int(head_data[1])), (int(head_data[2]), int(head_data[3])),
                              HEAD_REC_COLOUR)

                for joint_id in range(0, 16):
                    # Draw points
                    cv2.line(img, (int(joint_data[str(joint_id)][0]), int(joint_data[str(joint_id)][1])),
                             (int(joint_data[str(joint_id)][0]), int(joint_data[str(joint_id)][1])),
                             joint_colour(visible_joint[str(joint_id)]), 5)

                    # Draw Text to label the joint.
                    cv2.putText(img, org=(int(joint_data[str(joint_id)][0]) + TEXT_POSITION_MODIFIER_X,
                                          int(joint_data[str(joint_id)][1]) + TEXT_POSITION_MODIFIER_Y),
                                color=TEXT_COLOUR, text=str(joint_id), fontFace=cv2.FONT_HERSHEY_PLAIN,
                                fontScale=TEXT_SCALE,
                                thickness=TEXT_THICKNESS)

                    if joint_id < 15:
                        draw_limb(img, joint_data[str(joint_id)], joint_data[str(joint_id + 1)], joint_id)
                    else:
                        draw_limb(img, joint_data[str(joint_id)], joint_data[str(joint_id - 1)], joint_id)

            # Write the image to file
            cv2.imwrite(IMAGE_OUTPUT_PATH + image_file_name, img)


def draw_limb(img, start_point, end_point, joint_id):
    if joint_id != 9 and joint_id != 5:
        cv2.line(img, (int(start_point[0]), int(start_point[1])), (int(end_point[0]), int(end_point[1])),
                 color=(255, 255, 255), thickness=1)

