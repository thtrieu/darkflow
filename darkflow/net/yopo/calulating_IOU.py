import math

import cv2
import pyclipper


class Point:

    def __init__(self, x, y):
        self.x = int(x)
        self.y = int(y)


class Rectangle:

    def __init__(self, x, y, w, h, angle):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.angle = angle

    def draw(self, image, colour=(140, 140, 140)):
        # cv2.rectangle(image, (int(self.x - self.w / 2), int(self.y + self.h / 2)),
        #               (int(self.x + self.w / 2), int(self.y - self.h / 2)), (255, 0, 255), 1)
        pts = self.get_vertices_points()
        draw_polygon(image, pts, colour)

    def rotate_rectangle(self, theta):
        pt0, pt1, pt2, pt3 = self.get_vertices_points()

        # Point 0
        rotated_x = math.cos(theta) * (pt0.x - self.x) - math.sin(theta) * (pt0.y - self.y) + self.x
        rotated_y = math.sin(theta) * (pt0.x - self.x) + math.cos(theta) * (pt0.y - self.y) + self.y
        point_0 = Point(rotated_x, rotated_y)

        # Point 1
        rotated_x = math.cos(theta) * (pt1.x - self.x) - math.sin(theta) * (pt1.y - self.y) + self.x
        rotated_y = math.sin(theta) * (pt1.x - self.x) + math.cos(theta) * (pt1.y - self.y) + self.y
        point_1 = Point(rotated_x, rotated_y)

        # Point 2
        rotated_x = math.cos(theta) * (pt2.x - self.x) - math.sin(theta) * (pt2.y - self.y) + self.x
        rotated_y = math.sin(theta) * (pt2.x - self.x) + math.cos(theta) * (pt2.y - self.y) + self.y
        point_2 = Point(rotated_x, rotated_y)

        # Point 3
        rotated_x = math.cos(theta) * (pt3.x - self.x) - math.sin(theta) * (pt3.y - self.y) + self.x
        rotated_y = math.sin(theta) * (pt3.x - self.x) + math.cos(theta) * (pt3.y - self.y) + self.y
        point_3 = Point(rotated_x, rotated_y)

        return point_0, point_1, point_2, point_3

    def get_vertices_points(self):
        x0, y0, width, height, _angle = self.x, self.y, self.w, self.h, self.angle
        b = math.cos(math.radians(_angle)) * 0.5
        a = math.sin(math.radians(_angle)) * 0.5
        pt0 = Point(int(x0 - a * height - b * width), int(y0 + b * height - a * width))
        pt1 = Point(int(x0 + a * height - b * width), int(y0 - b * height - a * width))
        pt2 = Point(int(2 * x0 - pt0.x), int(2 * y0 - pt0.y))
        pt3 = Point(int(2 * x0 - pt1.x), int(2 * y0 - pt1.y))
        pts = [pt0, pt1, pt2, pt3]
        return pts

    def find_intersection_shape_area(self, clip_rectangle, vertices=False):

        point_0_rec1, point_1_rec1, point_2_rec1, point_3_rec1 = self.get_vertices_points()
        point_0_rec2, point_1_rec2, point_2_rec2, point_3_rec2 = clip_rectangle.get_vertices_points()

        subj = (
            ((point_0_rec1.x, point_0_rec1.y), (point_1_rec1.x, point_1_rec1.y), (point_2_rec1.x, point_2_rec1.y),
             (point_3_rec1.x, point_3_rec1.y))
        )

        if len(set(subj)) != 4 or int(self.w) == 0 or int(self.h) == 0 or int(clip_rectangle.w) == 0 or int(clip_rectangle.h) == 0:
            if vertices:
                return 0, []
            else:
                return 0

        clip = ((point_0_rec2.x, point_0_rec2.y), (point_1_rec2.x, point_1_rec2.y), (point_2_rec2.x, point_2_rec2.y),
                (point_3_rec2.x, point_3_rec2.y))

        if len(set(clip)) != 4:
            if vertices:
                return 0, []
            else:
                return 0


        pc = pyclipper.Pyclipper()
        pc.AddPath(clip, pyclipper.PT_CLIP, True)
        pc.AddPath(subj, pyclipper.PT_SUBJECT, True)

        solutions = pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)

        if len(solutions) == 0:
            if vertices:
                return 0, []
            else:
                return 0

        solution = solutions[0]

        pts = list(map(lambda i: Point(i[0], i[1]), solution))

        if vertices:
            return self._area(solution), pts

        return self._area(solution)

    def find_union_shape_area(self, clip_rectangle, vertices=False):

        point_0_rec1, point_1_rec1, point_2_rec1, point_3_rec1 = self.get_vertices_points()
        point_0_rec2, point_1_rec2, point_2_rec2, point_3_rec2 = clip_rectangle.get_vertices_points()

        subj = (
            ((point_0_rec1.x, point_0_rec1.y), (point_1_rec1.x, point_1_rec1.y), (point_2_rec1.x, point_2_rec1.y),
             (point_3_rec1.x, point_3_rec1.y))
        )

        if len(set(subj)) != 4 or int(self.w) == 0 or int(self.h) == 0 or int(clip_rectangle.w) == 0 or int(clip_rectangle.h) == 0:
            if vertices:
                return 0, []
            else:
                return 0

        clip = ((point_0_rec2.x, point_0_rec2.y), (point_1_rec2.x, point_1_rec2.y), (point_2_rec2.x, point_2_rec2.y),
                (point_3_rec2.x, point_3_rec2.y))

        if len(set(clip)) != 4:
            if vertices:
                return 0, []
            else:
                return 0

        pc = pyclipper.Pyclipper()
        pc.AddPath(clip, pyclipper.PT_CLIP, True)
        pc.AddPath(subj, pyclipper.PT_SUBJECT, True)

        solutions = pc.Execute(pyclipper.CT_UNION, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)
        solution = solutions[0]

        pts = list(map(lambda i: Point(i[0], i[1]), solution))

        if vertices:
            return self._area(solution), pts

        return self._area(solution)

    # Green's Theorem
    def _area(self, p):
        return 0.5 * abs(sum(x0 * y1 - x1 * y0
                             for ((x0, y0), (x1, y1)) in self._segments(p)))

    def _segments(self, p):
        return zip(p, p[1:] + [p[0]])

    def __str__(self):
        return "Rectangle: x: {}, y: {}, w: {}, h: {}, angle: {}".format(self.x, self.y, self.w, self.h, self.angle)

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


def intersection_over_union(bbox_ground_truth, bbox_predicted):
    """
    Calculates the area of overlap between the predicted bounding box and
    the ground-truth bounding box, also know as IOU
    :param bbox_ground_truth: YOPO Rectangle Object
    :param bbox_predicted: YOPO Rectangle Object
    :return: Intersection over union value
    """

    intersection = bbox_ground_truth.find_intersection_shape_area(bbox_predicted, vertices=False)
    union = bbox_ground_truth.find_union_shape_area(bbox_predicted, vertices=False)

    if union == 0:
        return 0

    return intersection / union


def show_image(img):
    cv2.namedWindow("Display window", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Display Window", img)
    cv2.waitKey(0)

if __name__ == "__main__":
    print("Calculate IOU of left-lower-arm")

    img = cv2.imread("/home/richard/git/darkflow/sub_set/images/040967287.jpg")

    #  Ground Truth - left-lower-arm
    centre_point_x = (1503 + 1553) / 2
    centre_point_y = (817 + 978) / 2
    print("x: {}, y: {}".format(centre_point_x, centre_point_y))
    angle = -72
    width = 50
    height = (978 - 817)

    rec1 = Rectangle(centre_point_x, centre_point_y, width, height, angle)
    # rec1.draw(img, colour=(0, 0, 250))

    # 2nd Rectangle - Predicted Box
    rec2_x = centre_point_x + 25
    rec2_y = centre_point_y + 25
    rec2_w = 50
    rec2_h = height
    rec2_angle = 30
    rec2 = Rectangle(rec2_x, rec2_y, rec2_w, rec2_h, rec2_angle)
    # rec2.draw(img, colour=(0, 0, 255))

    rec1 = Rectangle(599.0000009536743,  397.9999977350235,  105.99999847437786,  49.99999667005966,  -129.2513427734375)
    rec2 = Rectangle(490.2878352999687,  271.58785675652325,  0.2539491610125122,  2.3729914290343963,  -0.15401481091976166)
    # rec2 = Rectangle(100, 100, 0, 58.1619032498341, -86.7201919555664)




    # Calculate IOU
    intersection, shape_vertices = rec1.find_intersection_shape_area(rec2, vertices=True)
    # draw_polygon(img, shape_vertices)

    union, shape_vertices_union = rec1.find_union_shape_area(rec2, vertices=True)
    draw_polygon(img, shape_vertices_union, colour=(255, 255, 255))
    draw_polygon(img, rec1.get_vertices_points(), colour=(255, 0, 255))
    draw_polygon(img, rec2.get_vertices_points(), colour=(255, 0, 255))


    iou = intersection_over_union(rec1, rec2)

    print("IOU: {}".format(iou))
    cv2.imwrite("output_union.jpg", img)

    # Show result
    cv2.namedWindow("Display window", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Display Window", img)
    cv2.waitKey(0)