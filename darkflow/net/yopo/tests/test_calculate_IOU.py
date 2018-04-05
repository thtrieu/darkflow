import unittest
import cv2
import numpy as np
import math


from darkflow.net.yopo.calulating_IOU import Rectangle, intersection_over_union,draw_polygon, show_image


class TestCalculateIOU(unittest.TestCase):

    # Testing Basic Functionality

    def test_equal_IOU(self):

        rec1 = Rectangle(100, 100, 50, 50, 0)
        rec2 = Rectangle(100, 100, 50, 50, 0)

        iou = rec1.find_intersection_shape_area(rec2)


        self.assertTrue(1, iou)

    def test_equal_IOU_angled(self):

        rec1 = Rectangle(100, 100, 50, 50, 66)
        rec2 = Rectangle(100, 100, 50, 50, 66)

        iou = rec1.find_intersection_shape_area(rec2)

        self.assertTrue(1, iou)


    def test_find_union_shape_area(self):

        rec1 = Rectangle(100, 100, 50, 50, 0)
        rec2 = Rectangle(100, 100, 50, 50, 0)

        union_area = rec1.find_union_shape_area(rec2)

        # 50^2 -> 2500
        self.assertTrue(2500, union_area)

    #     todo intersection_over_union

    def test_intersection_over_union(self):

        rec1 = Rectangle(100, 100, 50, 50, 0)
        rec2 = Rectangle(100, 100, 50, 50, 0)

        iou = intersection_over_union(rec1, rec2)

        self.assertTrue(1 == iou, "Intersection Over Union basic test")

    def test_intersection_over_union_angled(self):

        rec1 = Rectangle(100, 100, 50, 50, 11)
        rec2 = Rectangle(100, 100, 50, 50, 11)

        iou = intersection_over_union(rec1, rec2)

        self.assertTrue(1 == iou, "Intersection Over Union basic test")


    # Edge Cases

    def test_zero_width_handling(self):

        rec1 = Rectangle(100, 100, 50, 50, 66)
        rec2 = Rectangle(100, 100, 0, 50, 66)

        iou = rec1.find_intersection_shape_area(rec2)

        self.assertTrue(0 == iou)

    def test_zero_height_handling(self):

        rec1 = Rectangle(100, 100, 50, 50, 66)
        rec2 = Rectangle(100, 100, 50, 0, 66)

        iou = rec1.find_intersection_shape_area(rec2)

        self.assertTrue(0 == iou)

    def test_zero_height_and_width_handling(self):

        rec1 = Rectangle(100, 100, 50, 50, 66)
        rec2 = Rectangle(100, 100, 0, 0, 66)

        iou = rec1.find_intersection_shape_area(rec2)

        self.assertTrue(0 == iou)

    def test_line_area_handling(self):
        '''
        We do not want a line we only want shapes that we can find areas of such as triangles and rectangles.
        '''

        rec1 = Rectangle(100, 100, 0, 50, 0)
        rec2 = Rectangle(100, 100, 0, 50, 0)

    # def test_triangle_intersect(self):
    #     '''
    #     Test if correct intersection area is return if a rectangle intersection a 2nd rectangle at 45 degree angle,
    #     such that it's intersection shape create a triangle.
    #     '''
    #
    #     # Blank Image
    #     img = np.zeros((200, 200, 3), np.uint8)
    #
    #     rec1 = Rectangle(100, 100, 50, 100, 0)
    #     rec2 = Rectangle(100, 100, 50, 50, 45)
    #
    #     draw_polygon(img, rec1.get_vertices_points(),)
    #     draw_polygon(img, rec2.get_vertices_points(), (255, 255, 255))
    #
    #     area, pts = rec1.find_intersection_shape_area(rec2, True)
    #
    #     # draw_polygon(img, pts, (0, 0, 255))
    #
    #     show_image(img)
    #
    #
    #
    #     print(area)
    #
    #
    #     self.assertEqual(12.5 , area)







if __name__ == '__main__':
    unittest.main()
