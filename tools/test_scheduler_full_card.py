"""Full-card reservations skip cards given 0 slots (shared hosts).

Run: PYTHONPATH=tools python3 -m unittest tools/test_scheduler_full_card.py
"""
import os
import unittest
from unittest import mock

import notebook_scheduler as ns


def scheduler(slots_per):
    env = {"D2L_GPU_SLOTS_PER": slots_per, "D2L_GPU_VRAM_PER": "",
           "D2L_CPU_SLOTS": "1", "D2L_GPU_MIB_PER_SLOT": "7680"}
    with mock.patch.dict(os.environ, env), \
            mock.patch.object(ns, "_detect", lambda key: ""):
        return ns.Scheduler(args=None)


class FullCardTest(unittest.TestCase):
    def item(self):
        return ns.Item("pytorch", "chapter_x/full.ipynb", None, None,
                       ("gpu", 1, "all"))

    def test_zero_slot_cards_are_not_free(self):
        s = scheduler("0,0,0,3")
        asg = s._reserve(self.item())
        self.assertEqual(asg["_"][1], [3])
        self.assertEqual(s.gpu_free, [0, 0, 0, 0])

    def test_no_card_with_slots_waits(self):
        s = scheduler("0,0,0,0")
        self.assertIsNone(s._reserve(self.item()))


if __name__ == "__main__":
    unittest.main()
