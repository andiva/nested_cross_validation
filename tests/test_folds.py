import unittest
from nested_cross_validation.loops import uniform_loop


class NCVTestCases(unittest.TestCase):

    def test_uniform_outer_loop(self):
        inds = list(uniform_loop(0, 99, 5, 4))
        expected_inds = [([0, 44], [45, 55]), ([11, 55], [56, 66]),
                         ([22, 66], [67, 77]), ([33, 77], [78, 88]),
                         ([44, 88], [89, 99])]
        self.assertEqual(inds, expected_inds)

    def test_uniform_inner_loop(self):
        inds = list(uniform_loop(22, 66, 2, 1))
        expected_inds = [([22, 37], [38, 51]), ([37, 51], [52, 66])]
        self.assertEqual(inds, expected_inds)


if __name__ == '__main__':
    unittest.main()