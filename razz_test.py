import unittest
from razz import Razz

class TestRazz(unittest.TestCase):
    def setUp(self):
        self.razz = Razz()

    # *P1*: Player 1 wins
    def test_winner(self):
        # *P1* = Clean ace low, P2 = Pair of 5's
        self.razz.set_player_hands([1,7,20], [5,6,18])
        self.assertEqual(self.razz.winner(3), 0)
        # P1 = Pair of ace, *P2* = Clean 2 low
        self.razz.set_player_hands([0,4,5,13,26,39,15], [18,6,19,32,1,2,10])
        self.assertEqual(self.razz.winner(5), 1)
        # *P1* = Full house ace, P2 = Full house king
        self.razz.set_player_hands([0,13,26,39,1,14,27], [12,25,38,51,11,24,37])
        self.assertEqual(self.razz.winner(5), 0)
        # *P1* = ToaK 2's P2 = ToaK 4's
        self.razz.set_player_hands([1,14,27], [5,18,31])
        self.assertEqual(self.razz.winner(3), 0)
        # *P1* = Two pair 3-5, P2 = Two pair 4-5
        self.razz.set_player_hands([2,15,4,17], [3,16,30,43])
        self.assertEqual(self.razz.winner(4), 0)

        

if __name__ == "__main__":
     unittest.main()