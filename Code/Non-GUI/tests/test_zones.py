import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from keyforge.cards.card import Card
from keyforge.cards.card_data import get_card_def
from keyforge.zones import Deck, DiscardPile, PlayArea


class TestZones(unittest.TestCase):
    def test_deck_draw_order(self):
        cards = [Card(get_card_def("Urchin"), 1) for _ in range(3)]
        deck = Deck(cards)
        self.assertEqual(deck.draw_top(), cards[0])
        self.assertEqual(deck.draw_top(), cards[1])
        self.assertEqual(len(deck), 1)

    def test_deck_put_on_top(self):
        c1, c2 = Card(get_card_def("Urchin"), 1), Card(get_card_def("Old Bruno"), 1)
        deck = Deck([c1])
        deck.put_on_top(c2)
        self.assertEqual(deck.draw_top(), c2)

    def test_discard_stack_order(self):
        pile = DiscardPile()
        c1, c2 = Card(get_card_def("Urchin"), 1), Card(get_card_def("Old Bruno"), 1)
        pile.push(c1)
        pile.push(c2)
        self.assertEqual(pile.pop(), c2)
        self.assertEqual(pile.pop(), c1)

    def test_play_area_flank_and_neighbors(self):
        area = PlayArea()
        a, b, c = (Card(get_card_def("Urchin"), 1) for _ in range(3))
        area.add_creature(a)
        area.add_creature(b)
        area.add_creature(c, flank="left")
        self.assertEqual(area.creatures, [c, a, b])
        self.assertTrue(area.is_flank(c))
        self.assertTrue(area.is_flank(b))
        self.assertFalse(area.is_flank(a))
        self.assertEqual(area.neighbors(a), [c, b])


if __name__ == "__main__":
    unittest.main()
