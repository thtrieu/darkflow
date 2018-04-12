import unittest

from darkflow.net.yopo.network.calulating_IOU import Rectangle, intersection_over_union, check_intersection_is_line


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

        self.assertTrue(0 == iou, "test_zero_width_handling")

    def test_zero_height_handling(self):
        rec1 = Rectangle(100, 100, 50, 50, 66)
        rec2 = Rectangle(100, 100, 50, 0, 66)

        iou = rec1.find_intersection_shape_area(rec2)

        self.assertTrue(0 == iou, "zero_height_handling")

    def test_zero_height_and_width_handling(self):
        rec1 = Rectangle(100, 100, 50, 50, 66)
        rec2 = Rectangle(100, 100, 0, 0, 66)

        iou = rec1.find_intersection_shape_area(rec2)

        self.assertTrue(0 == iou, "test_zero_height_and_width_handling")

    def test_floating_point_area_less_than_one(self):
        rec1 = Rectangle(752.5, 392.0000011580331, 131.0000069588125, 49.999997442956285, 90.0)
        rec2 = Rectangle(733.6542248725891, 368.1448962858745, 0.9, 0.9, 90.0)
        iou = intersection_over_union(rec1, rec2)

        self.assertTrue(0 == iou)

    def test_floating_point_area_less_than_one_different_angles(self):
        rec1 = Rectangle(752.5, 392.0000011580331, 131.0000069588125, 49.999997442956285, 90.0)
        rec2 = Rectangle(733.6542248725891, 368.1448962858745, 0.9, 0.9, 66.87)
        iou = intersection_over_union(rec1, rec2)

        self.assertTrue(0 == iou)

    def test_line_area_handling(self):
        '''
        We do not want a line we only want shapes that we can find areas of such as triangles and rectangles.
        '''

        gt_rec1 = Rectangle(100, 100, 0, 50, 0)
        netout_rec2 = Rectangle(100, 100, 0, 50, 0)

    def test_network_output_iou_calculation(self):
        '''
        This is using an example from the network
        '''
        gt_rec1 = Rectangle(745.0, 526.000000834465, 136.00000770980955, 49.999997442956285, 96.29394292831421)
        netout_rec2 = Rectangle(695.4430913925171, 490.2400955557823, 95.81683375372563, 52.13581478280169,
                                91.01072072982788)

        iou = intersection_over_union(gt_rec1, netout_rec2)

        self.assertTrue(0.0016777116013757234 == iou)

    # todo fix name edge case test
    def test(self):
        """
        Edge Case Test -
        :return:
        """
        gt_rec = Rectangle(658.4999997275216, 220.00000068119596, 89.00000038378948, 49.999997442956285,
                           -78.43986690044403)
        netout_rec = Rectangle(646.7366170883179, 212.04363265207837, 1.3590652958357197, 1.0475772630980373,
                               -41.93526774644852)

        iou = intersection_over_union(gt_rec, netout_rec)

        self.assertEqual(0, iou, "Testing a known broken input that should catch and return to 0")

    def test_points_handel_line(self):

        """
        Tests check_intersection_is_line - Checks if the shape is a line and is so returns True
        """

        zero_width = check_intersection_is_line(Rectangle(4, 4, 0, 2, 0))
        zero_height = check_intersection_is_line(Rectangle(4, 4, 2, 0, 0))
        rec = Rectangle(4, 4, 0, 2, -66)
        zero_width_postive_angle = check_intersection_is_line(rec)
        zero_height_negative_angle = check_intersection_is_line(Rectangle(4, 4, 2, 0, 99))

        self.assertTrue(zero_width, "zero_width")
        self.assertTrue(zero_height, "zero_height")
        self.assertTrue(zero_width_postive_angle, "zero_width_postive_angle")
        self.assertTrue(zero_height_negative_angle, "zero_height_negative_angle")

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
    #     rec2 = Rectangle(100, 100, 50, 50, 45)s
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
