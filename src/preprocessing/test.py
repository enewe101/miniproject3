import unittest
import preprocessing as p
import numpy as np

class TestKMeansFinder(unittest.TestCase):

	def test_distance(self):

		points = [
			[1,2,3],
			[5,6,7]
		]
		clusterer = p.KMeansFinder()

		found_distance = clusterer.calc_distance(points[0], points[1])
		expected_distance = np.sqrt(3 * (4**2))

		self.assertEqual(found_distance, expected_distance)


	def test_clustering(self):
		near_points = [
			[0,0,0,0],
			[1,1,1,1],
			[2,2,2,2]
		]

		far_points = [
			[8,8,8,8],
			[9,9,9,9],
			[10,10,10,10],
		]

		expected_means = np.mean(near_points,0), np.mean(far_points,0)

		all_points = near_points + far_points
		clusterer = p.KMeansFinder()
		found_means = clusterer.cluster(all_points, 2)

		expected_means = [tuple(x) for x in expected_means]
		found_means = [tuple(x) for x in found_means]

		self.assertItemsEqual(expected_means, found_means)



if __name__ == '__main__':
	unittest.main()

