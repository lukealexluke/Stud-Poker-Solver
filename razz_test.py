import unittest
from razz import Razz

class TestRazz(unittest.TestCase):
    def setUp(self):
        self.game = Razz()

    def test_best_hand(self):
        self.assertEqual(self.game.best_hand([[0, 1, 2, 3, 4, 5, 6],[0]], 0, 5),[0,1,2,3,4])
        self.assertEqual(self.game.best_hand([[14, 15, 16, 17, 23, 36, 49],[0]], 0, 5),[1,2,3,4,10])
        self.assertEqual(self.game.best_hand([[12, 25, 38, 51, 11, 24, 37],[0]], 0, 5),[11,11,11,12,12])        
        

if __name__ == "__main__":
     unittest.main()