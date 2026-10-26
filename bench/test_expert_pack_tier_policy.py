# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the SSD-LLaMA three-tier placement policy (fake backend)."""

from __future__ import annotations

import random
import unittest

from sglang.srt.layers.moe.expert_pack_tier import RAM, SSD, VRAM, TierPolicy


class FakeBackend:
    """Simulates the store's acquire() bookkeeping on top of TierPolicy without
    any buffers: tracks the legacy counters the same way _acquire_tiered does
    and marks RAM slots busy for a configurable number of steps."""

    def __init__(self, vram, ram, **kw):
        self.p = TierPolicy(vram, ram, **kw)
        self.cache_hits = self.cache_misses = self.cache_evictions = 0
        self.busy_until = {}  # ram slot -> step until which it is "in flight"
        self.d2h_copies = 0

    def evictable(self, slot):
        return self.busy_until.get(slot, -1) < self.p.step

    def acquire(self, layer, experts, busy_steps=0):
        p = self.p
        if layer == 0:
            p.begin_step()
        requested = {(layer, e) for e in experts}
        for key in requested:
            p.record_access(key)
        outcome = {}
        for key in sorted(requested):
            where = p.classify(key)
            outcome[key] = where
            if where == VRAM:
                self.cache_hits += 1
                continue
            self.cache_misses += 1
            retained = p.should_retain(key)  # decided before a slot is freed
            idx = p.pick_vram_victim(requested, retained=retained)
            old, result = p.evict_vram(idx, can_d2h=p.ram_slots > 0)
            if old is not None:
                self.cache_evictions += 1
            if result == "d2h":
                self.d2h_copies += 1
                self.busy_until[p.ram_slot_of(old)] = p.step + busy_steps - 1
            p.install_vram(idx, key, retained=retained)
            if where == SSD and p.ram_slots:
                slot, _ = p.ram_alloc(key, self.evictable)
                assert slot is not None, "backpressure case not modelled here"
                self.busy_until[slot] = p.step + busy_steps - 1
        p.check_invariants()
        return outcome


def _reference_victim(p, protected):
    """Brute-force scan the heap must reproduce exactly (incl. slot-index tie-break)."""
    best = None
    for idx, e in enumerate(p.vram):
        if e.key is None or not e.retained or e.key in protected:
            continue
        h = p.history[e.key]
        score = (h.access_count, h.last_access, idx)
        if best is None or score < best[0]:
            best = (score, idx)
    return None if best is None else best[1]


class TestHeapAndPool(unittest.TestCase):
    def test_heap_victim_equals_full_scan_with_slot_tiebreak(self):
        rng = random.Random(7)
        p = TierPolicy(12, 0, policy="paper", admission_min_accesses=2, admission_window_steps=3)
        for step in range(600):
            p.begin_step()
            keys = {(0, rng.randrange(20)) for _ in range(rng.randrange(1, 5))}
            for k in keys:
                p.record_access(k)
            for k in sorted(keys):
                if p.classify(k) == VRAM:
                    continue
                retained = p.should_retain(k)
                if retained and not p.has_free_vram():
                    self.assertEqual(p._heap_min_retained(keys), _reference_victim(p, keys))
                idx = p.pick_vram_victim(keys, retained=retained)
                p.evict_vram(idx, can_d2h=False)
                p.install_vram(idx, k, retained=retained)
            p.check_invariants()  # includes heap <-> retained-set consistency
        self.assertGreater(p.stats.vram_evictions, 100)

    def test_transient_pool_rotates_fifo_and_leaves_retained_untouched(self):
        p = TierPolicy(4, 0, policy="paper", transient_slots=3)
        self.assertEqual(len(p.vram), 7)
        # fill the retained region with hot experts
        for _ in range(3):
            p.begin_step()
            for k in [(0, 1), (0, 2), (0, 3), (0, 4)]:
                p.record_access(k)
                if p.classify(k) != VRAM:
                    i = p.pick_vram_victim({(0, 1), (0, 2), (0, 3), (0, 4)}, retained=True)
                    p.install_vram(i, k, retained=True)
        retained_before = {e.key for e in p.vram[:4]}
        self.assertFalse(p.has_free_vram())
        # six cold experts, one per update: pool slots 4,5,6 are used in rotation
        got = []
        for e in range(10, 16):
            p.begin_step(); p.record_access((0, e)); p.classify((0, e))
            self.assertFalse(p.should_retain((0, e)))
            i = p.pick_vram_victim({(0, e)}, retained=False)
            p.evict_vram(i, can_d2h=False); p.install_vram(i, (0, e), retained=False)
            got.append(i)
        self.assertEqual(got, [4, 5, 6, 4, 5, 6])
        self.assertEqual({e.key for e in p.vram[:4]}, retained_before)  # hot set untouched
        self.assertEqual(p.stats.transient_uses, 6)
        p.check_invariants()

    def test_pool_exhausted_within_update_falls_back_to_retained_region(self):
        p = TierPolicy(3, 0, policy="paper", transient_slots=1)
        for k in [(0, 1), (0, 2), (0, 3)]:
            p.begin_step(); p.record_access(k); p.classify(k)
            p.install_vram(p.pick_vram_victim({k}, retained=True), k, retained=True)
        p.begin_step()
        req = {(0, 7), (0, 8)}
        for k in sorted(req):
            p.record_access(k)
        slots = []
        for k in sorted(req):
            p.classify(k)
            i = p.pick_vram_victim(req, retained=False)
            p.evict_vram(i, can_d2h=False); p.install_vram(i, k, retained=False)
            slots.append(i)
        self.assertEqual(slots[0], 3)            # pool slot first
        self.assertLess(slots[1], 3)             # then a retained-region victim (lowest score)
        p.check_invariants()

    def test_pool_requires_paper_policy(self):
        with self.assertRaises(ValueError):
            TierPolicy(2, 0, policy="lru", transient_slots=2)


class TestTierPolicy(unittest.TestCase):
    def test_lookup_order_vram_then_ram_then_ssd(self):
        b = FakeBackend(vram=1, ram=4, policy="paper")
        self.assertEqual(b.acquire(0, [7])[(0, 7)], SSD)      # cold -> SSD
        self.assertEqual(b.acquire(0, [7])[(0, 7)], VRAM)     # resident in VRAM
        b.acquire(0, [8])                                      # evicts 7 from the 1-slot VRAM
        self.assertEqual(b.acquire(0, [7])[(0, 7)], RAM)      # 7 still in RAM -> RAM hit
        s = b.p.stats
        self.assertEqual((s.vram_hits, s.ram_hits, s.ssd_misses), (1, 1, 2))

    def test_counter_identities(self):
        rng = random.Random(1)
        b = FakeBackend(vram=6, ram=5, policy="paper", admission_min_accesses=2, admission_window_steps=4)
        for _ in range(400):
            b.acquire(rng.randrange(2), rng.sample(range(12), 3))
        s = b.p.stats
        self.assertEqual(b.cache_hits, s.vram_hits)
        self.assertEqual(b.cache_misses, s.ram_hits + s.ssd_misses)
        self.assertEqual(b.cache_evictions, s.vram_evictions)
        self.assertEqual(s.demotions_to_ram, s.demotions_resident + s.demotions_d2h)
        self.assertEqual(s.vram_evictions, s.demotions_to_ram + s.demotions_discarded)

    def test_transient_admission_protects_hot_expert(self):
        # VRAM has 2 slots, both hot; a single cold access must not evict them.
        b = FakeBackend(vram=2, ram=0, policy="paper", admission_min_accesses=2, admission_window_steps=8)
        for _ in range(3):
            b.acquire(0, [1, 2])
        self.assertTrue(all(e.retained for e in b.p.vram))
        self.assertEqual(b.acquire(0, [3])[(0, 3)], SSD)      # cold; VRAM full -> transient
        self.assertEqual(b.p.stats.transient_uses, 1)
        transient_slot = b.p.vram_slot_of((0, 3))
        self.assertFalse(b.p.vram[transient_slot].retained)
        # the next cold expert evicts the transient one, not a hot one
        b.acquire(0, [4])
        self.assertNotIn((0, 3), b.p.key_to_vram)
        self.assertEqual(b.p.vram_slot_of((0, 4)), transient_slot)
        self.assertEqual(len(b.p.key_to_vram), 2)
        self.assertTrue({(0, 1), (0, 2)} & set(b.p.key_to_vram))  # a hot expert survived
        # 4 hit again in its transient slot: not promoted in place (paper: temporary
        # slot), but its history now qualifies it for retention on its next load
        self.assertEqual(b.acquire(0, [4])[(0, 4)], VRAM)
        self.assertFalse(b.p.vram[b.p.vram_slot_of((0, 4))].retained)
        self.assertTrue(b.p.should_retain((0, 4)))
        self.assertEqual(b.p.stats.promotions, 0)

    def test_free_vram_always_retains(self):
        b = FakeBackend(vram=3, ram=0, policy="paper")
        b.acquire(0, [5])
        self.assertTrue(b.p.vram[b.p.vram_slot_of((0, 5))].retained)
        self.assertEqual(b.p.stats.transient_uses, 0)

    def test_vram_eviction_order_lowest_count_then_oldest(self):
        p = TierPolicy(3, 0, policy="paper")
        for step, keys in enumerate([[(0, 1), (0, 2), (0, 3)], [(0, 1), (0, 2)], [(0, 1)]]):
            p.begin_step()
            for k in keys:
                p.record_access(k)
                if p.classify(k) != VRAM:
                    i = p.pick_vram_victim(set(keys))
                    p.evict_vram(i, can_d2h=False)
                    p.install_vram(i, k, retained=True)
        # counts: 1->3, 2->2, 3->1 ; victim must be (0,3)
        self.assertEqual(p.vram[p.pick_vram_victim(set())].key, (0, 3))
        # tie on count: older last_access loses
        p.begin_step(); p.record_access((0, 3)); p.classify((0, 3))   # 3 -> count 2, last=4 ; 2 -> count 2, last=2
        self.assertEqual(p.vram[p.pick_vram_victim(set())].key, (0, 2))
        # protected working set is never chosen
        self.assertEqual(p.vram[p.pick_vram_victim({(0, 2)})].key, (0, 3))

    def test_demotion_resident_d2h_discard(self):
        p = TierPolicy(1, 1, policy="paper")
        p.begin_step(); p.record_access((0, 1)); p.classify((0, 1))
        p.install_vram(0, (0, 1), retained=True)
        slot, _ = p.ram_alloc((0, 1), lambda s: True)       # bytes land in RAM (unified design)
        self.assertEqual(p.evict_vram(0, can_d2h=True), ((0, 1), "resident"))
        p.begin_step(); p.record_access((0, 2)); p.classify((0, 2))
        p.install_vram(0, (0, 2), retained=True)              # 2 came from SSD but its RAM slot was taken by 1 -> LRU evicts 1
        s2, ev = p.ram_alloc((0, 2), lambda s: True)
        self.assertEqual(ev, (0, 1))
        self.assertEqual(p.stats.ram_evictions, 1)
        p.begin_step(); p.record_access((0, 3)); p.classify((0, 3))
        # evicting 2 from VRAM: its RAM copy is resident -> free demotion
        self.assertEqual(p.evict_vram(0, can_d2h=True)[1], "resident")
        p.install_vram(0, (0, 3), retained=True)
        # RAM full with 2 (not busy) and no free slot: evicting 3 -> discard (no D2H into an occupied slot)
        self.assertEqual(p.evict_vram(0, can_d2h=True)[1], "discard")
        # with a free RAM slot the eviction reserves it for a D2H copy
        p2 = TierPolicy(1, 2, policy="paper")
        p2.begin_step(); p2.record_access((0, 9)); p2.classify((0, 9)); p2.install_vram(0, (0, 9), retained=True)
        self.assertEqual(p2.evict_vram(0, can_d2h=True)[1], "d2h")
        self.assertIsNotNone(p2.ram_slot_of((0, 9)))
        self.assertEqual(p2.stats.demotions_d2h, 1)
        p2.check_invariants()

    def test_ram_lru_and_inflight_protection(self):
        p = TierPolicy(8, 2, policy="paper")
        busy = set()
        a, _ = p.ram_alloc((0, 1), lambda s: s not in busy)
        b, _ = p.ram_alloc((0, 2), lambda s: s not in busy)
        p.ram_touch((0, 1))                                    # 2 is now LRU
        busy.add(b)                                            # ... but in flight
        c, evicted = p.ram_alloc((0, 3), lambda s: s not in busy)
        self.assertEqual((c, evicted), (a, (0, 1)))            # LRU-but-busy skipped, next LRU evicted
        busy.add(c)
        self.assertEqual(p.ram_alloc((0, 4), lambda s: s not in busy), (None, None))  # backpressure
        busy.clear()
        d, evicted = p.ram_alloc((0, 4), lambda s: s not in busy)
        self.assertEqual(evicted, (0, 2))                      # plain LRU once idle
        self.assertEqual(p.stats.ram_evictions, 2)

    def test_lru_policy_matches_legacy_victim_rule(self):
        p = TierPolicy(2, 0, policy="lru")
        for k in [(0, 1), (0, 2)]:
            p.begin_step(); p.record_access(k); p.classify(k)
            p.install_vram(p.pick_vram_victim({k}), k, retained=p.should_retain(k))
        p.begin_step(); p.record_access((0, 1)); p.classify((0, 1))   # 1 is more frequent
        self.assertTrue(p.should_retain((0, 9)))                       # lru never uses transient slots
        self.assertEqual(p.vram[p.pick_vram_victim(set())].key, (0, 2))
        self.assertEqual(p.stats.transient_uses, 0)

    def test_bad_config(self):
        with self.assertRaises(ValueError):
            TierPolicy(1, 0, policy="mru")
        with self.assertRaises(ValueError):
            TierPolicy(0, 0)


if __name__ == "__main__":
    unittest.main()
