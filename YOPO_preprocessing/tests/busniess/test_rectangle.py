import unittest

from src.busniess.calulating_IOU import Rectangle


class TestRectangle(unittest.TestCase):

    def test_find_intersection_shape_area(self):

        # Test Rectangle
        w = 10
        h = 30
        x = 50
        y = 50

        angles = [0, 40, 80, 120, 160, 200, 240, 280, 320, 360, -40, -80, -120, -160, -200, -240, -280, -320, -360]
        bbox = Rectangle(x, y, w, h, angles[0])

        for a in angles:
            bbox.angle = a
            bbox_points = bbox.get_vertices_points()
            # print(bbox_points)
            pts = list(map(lambda i: [i.x, i.y], bbox_points))
            bbox_area = bbox._area(pts)

            bbox.rotate_rectangle(50)
            bbox_points = bbox.get_vertices_points()
            # print(bbox_points)
            pts = list(map(lambda i: [i.x, i.y], bbox_points))
            bbox_area_rotated = bbox._area(pts)

            print("Rotation angle: {}".format(a))
            print("bbox area: {}".format(bbox_area))
            print("bbox rotated area: {}".format(bbox_area_rotated))
            print("--------------------------------------")

            self.assertEqual(bbox_area, bbox_area_rotated)


if __name__ == '__main__':
    unittest.main()
