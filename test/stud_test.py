import unittest
from stud import Game, SevenCardStud

class TestStud(unittest.TestCase):
    def setUp(self):
        self.game = SevenCardStud()

    # is_terminal function test
    def test_is_terminal(self):
        self.assertEqual(self.game.is_terminal([0,1,4]), True)
        self.assertEqual(self.game.is_terminal([0,1,2,'C',3,3,4]), True)
        self.assertEqual(self.game.is_terminal([0,1,2,'C',3,2,'C',3,2,'C',5,5,'C',5,5]), True)
        self.assertEqual(self.game.is_terminal([0,1,2,'C',5,5,'C',5,5,'C',5,5,'C',5,5]), True)
        self.assertEqual(self.game.is_terminal([0,1,2,'C',5,5,'C',5,5,'C',5,5,'C',5]), False)
        self.assertEqual(self.game.is_terminal([]), False)
        self.assertEqual(self.game.is_terminal([0]), False)

    def test_get_legal_actions(self):
        self.assertEqual(self.game.get_legal_actions([0]), [0, 1, 1, 0, 1, 0, 0])
        self.assertEqual(self.game.get_legal_actions([0,1]), [0, 0, 1, 1, 1, 0, 0])
        self.assertEqual(self.game.get_legal_actions([0,1,3,3]), [0, 0, 1, 0, 1, 0, 0])
        self.assertEqual(self.game.get_legal_actions([0,1,2,'C',3,3,3,3]), [0, 0, 1, 0, 1, 0, 0])
        


        

if __name__ == "__main__":
     unittest.main()