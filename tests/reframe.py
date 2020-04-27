from unittest import TestCase
from pprint import pprint
import numpy
from numpy.testing import assert_array_equal


class ReframeTests(TestCase):

    def test_center_smaller(self):
        from dfm.reframe import reframe
        a = numpy.array([
            [1, 1, 1],
            [1, 2, 1],
            [1, 1, 1]
        ])
        expected = numpy.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 2, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ])
        out = reframe(a, width=5, height=5, x=2, y=2)
        print('')
        pprint(out)
        assert_array_equal(out, expected)

    def test_center_larger(self):
        from dfm.reframe import reframe
        a = numpy.array([
            [1, 1, 1, 1, 1],
            [1, 2, 2, 2, 1],
            [1, 2, 3, 2, 1],
            [1, 2, 2, 2, 1],
            [1, 1, 1, 1, 1],
        ])
        expected = numpy.array([
            [2, 2, 2],
            [2, 3, 2],
            [2, 2, 2]
        ])
        out = reframe(a, width=3, height=3, x=1, y=1)
        print('')
        pprint(out)
        assert_array_equal(out, expected)

    def test_offcenter_smaller(self):
        from dfm.reframe import reframe
        a = numpy.array([
            [1, 1, 1],
            [1, 2, 1],
            [1, 1, 1]
        ])
        expected = numpy.array([
            [1, 1, 1, 0, 0],
            [1, 2, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ])
        out = reframe(a, width=5, height=5, x=1, y=1)
        print('')
        pprint(out)
        assert_array_equal(out, expected)

    def test_offcenter_larger(self):
        from dfm.reframe import reframe
        a = numpy.array([
            [1, 1, 1, 1, 1],
            [1, 2, 2, 2, 1],
            [1, 2, 3, 2, 1],
            [1, 2, 2, 2, 1],
            [1, 1, 1, 1, 1],
        ])
        expected = numpy.array([
            [2, 3, 2],
            [2, 2, 2],
            [1, 1, 1]
        ])
        out = reframe(a, width=3, height=3, x=1, y=0)
        print('')
        pprint(out)
        assert_array_equal(out, expected)

    def test_offcenter_smaller_cropped(self):
        from dfm.reframe import reframe
        a = numpy.array([
            [1, 1, 1],
            [1, 2, 1],
            [1, 1, 1]
        ])
        expected = numpy.array([
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 2],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ])
        out = reframe(a, width=5, height=5, x=4, y=1)
        print('')
        pprint(out)
        assert_array_equal(out, expected)

    def test_smaller_even_output(self):
        from dfm.reframe import reframe
        a = numpy.array([
            [1, 1, 1],
            [1, 2, 1],
            [1, 1, 1]
        ])
        expected = numpy.array([
            [0, 0, 0, 0],
            [0, 1, 1, 1],
            [0, 1, 2, 1],
            [0, 1, 1, 1],
        ])
        out = reframe(a, width=4, height=4, x=2, y=2)
        print('')
        pprint(out)
        assert_array_equal(out, expected)
