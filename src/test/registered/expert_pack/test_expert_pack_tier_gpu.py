# SPDX-License-Identifier: Apache-2.0
"""GPU integration test of the three-tier acquire() path on the synthetic pack."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from sglang.srt.layers.moe.expert_pack import ExpertPackStore
from test_expert_pack_runtime import _make_pack  # same directory


def _store(root, **kw):
    pack, manifest, digests = _make_pack(root)
    return ExpertPackStore(
        pack, manifest_path=manifest, expected_layers=1, expected_experts=2,
        expected_top_k=1, expected_source_sha256=digests["source"],
        expected_model_identity_sha256=digests["model_identity"],
        expected_config_sha256=digests["config"], cache_vram_mib=1,
        stage_slots=1, **kw)


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestTieredAcquireGPU(unittest.TestCase):
    def _run(self, store):
        # Force a 1-slot retained VRAM cache (plus the configured transient pool)
        # so experts 0 and 1 alternate.
        store._cache = None
        store.initialize_device_cache("cuda")
        pool = int(store.stats.get("transient_pool_slots", 0))
        n = store.stats["cache_capacity_experts"]
        keep = [0] + list(range(n, n + pool))
        store._cache = store._cache[keep].clone()
        store._cache_slots = [store._cache_slots[i] for i in keep]
        if store._tier is not None:
            tier = store._tier
            tier.vram = [tier.vram[i] for i in keep]
            tier._version = [0] * len(keep)
            tier.retained_slots = 1
        dev = torch.device("cuda")
        expected = {0: [0] * 17 + [1] * 17 + [2] * 17, 1: [3] * 17 + [4] * 17 + [5] * 17}
        seq = [0, 0, 1, 0, 1, 1, 0]
        for expert in seq:
            slots_t, slots = store.acquire(0, torch.tensor([[expert]], device=dev))
            torch.cuda.synchronize()
            self.assertEqual(store._cache[slots[0]].tolist(), expected[expert])
            store.mark_used(slots)
        snap = store.snapshot()
        store.close()
        return snap

    def test_paper_policy_with_host_cache_counts_and_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            s = _store(Path(d), host_cache_mib=1, vram_admission_policy="paper",
                       admission_min_accesses=2, admission_window_steps=8, transient_slots=0)
            snap = self._run(s)
        self.assertEqual(snap["transient_pool_slots"], 0)
        self.assertEqual(snap["cache_hits"], snap["vram_hits"])
        self.assertEqual(snap["cache_misses"], snap["ram_hits"] + snap["ssd_misses"])
        self.assertEqual(snap["cache_evictions"], snap["vram_evictions"])
        # 7 accesses on a 1-slot VRAM: first two are SSD misses, the rest resolve in RAM or VRAM
        self.assertEqual(snap["ssd_misses"], 2)
        self.assertGreater(snap["ram_hits"], 0)
        self.assertEqual(snap["pack_read_bytes"], snap["ssd_misses"] * s.object_payload_bytes)
        self.assertEqual(snap["h2d_bytes"], (snap["ram_hits"] + snap["ssd_misses"]) * s.object_payload_bytes)
        self.assertEqual(snap["io_errors"], 0)
        self.assertEqual(snap["ram_cache_bytes_used"], 2 * s.object_payload_bytes)
        self.assertIn("ram", snap["tier_h2d_latency_hist_us"])

    def test_legacy_path_untouched_when_disabled(self):
        with tempfile.TemporaryDirectory() as d:
            s = _store(Path(d))  # host_cache_mib=0, lru -> original code
            self.assertFalse(s.tiered)
            snap = self._run(s)
        self.assertEqual(snap["ram_hits"], 0)
        self.assertEqual(snap["vram_hits"], snap["cache_hits"])
        self.assertEqual(snap["ssd_misses"], snap["cache_misses"])

    def test_paper_policy_without_host_cache(self):
        with tempfile.TemporaryDirectory() as d:
            s = _store(Path(d), vram_admission_policy="paper", transient_slots=0)
            self.assertTrue(s.tiered)
            snap = self._run(s)
        self.assertEqual(snap["ram_hits"], 0)
        self.assertEqual(snap["cache_misses"], snap["ssd_misses"])
        self.assertEqual(snap["host_cache_slots"], 0)


    def test_transient_pool_on_top_of_budget(self):
        with tempfile.TemporaryDirectory() as d:
            s = _store(Path(d), host_cache_mib=1, vram_admission_policy="paper",
                       admission_min_accesses=2, admission_window_steps=8, transient_slots=2)
            s._cache = None
            s.initialize_device_cache("cuda")
            n = s.stats["cache_capacity_experts"]
            self.assertEqual(s.stats["transient_pool_slots"], 2)
            self.assertEqual(s._cache.shape[0], n + 2)              # pool is extra rows
            self.assertEqual(s.stats["cache_capacity_bytes"], n * s.object_payload_bytes)
            self.assertEqual(s._tier.retained_slots, n)
            snap = self._run(s)
        self.assertEqual(snap["cache_hits"], snap["vram_hits"])
        self.assertEqual(snap["cache_misses"], snap["ram_hits"] + snap["ssd_misses"])
        self.assertEqual(snap["h2d_bytes"], (snap["ram_hits"] + snap["ssd_misses"]) * s.object_payload_bytes)
        self.assertGreater(snap["transient_uses"], 0)


    def test_route_trace_record_and_oracle_prefetch_replay(self):
        import json
        seq = [0, 0, 1, 0, 1, 1, 0]
        expected = {0: [0] * 17 + [1] * 17 + [2] * 17, 1: [3] * 17 + [4] * 17 + [5] * 17}
        with tempfile.TemporaryDirectory() as d:
            trace = Path(d) / "route.jsonl"
            s = _store(Path(d), host_cache_mib=1, vram_admission_policy="paper", transient_slots=0,
                       route_trace_path=trace, update_trace_path=Path(d) / "upd.jsonl")
            self._run(s)
            lines = [json.loads(l) for l in trace.read_text().splitlines()]
            self.assertEqual([(r["s"], r["l"], r["e"]) for r in lines], [(i + 1, 0, [e]) for i, e in enumerate(seq)])
            upd = [json.loads(l) for l in (Path(d) / "upd.jsonl").read_text().splitlines()]
            self.assertEqual(len(upd), len(seq))
            self.assertTrue(all(u["t_out"] > u["t_in"] for u in upd))
            # replay with oracle prefetch one layer (= one step here) ahead
            s2 = _store(Path(d), host_cache_mib=1, vram_admission_policy="paper", transient_slots=0,
                        prefetch_trace_path=trace, prefetch_depth=1)
            s2._cache = None
            s2.initialize_device_cache("cuda")
            # keep 2 retained slots so the prefetched expert does not evict the current one
            dev = torch.device("cuda")
            for expert in seq:
                slots_t, slots = s2.acquire(0, torch.tensor([[expert]], device=dev))
                torch.cuda.synchronize()
                self.assertEqual(s2._cache[slots[0]].tolist(), expected[expert])
                s2.mark_used(slots)
            snap = s2.snapshot()
            s2.close()
        self.assertGreater(snap["prefetch_issued"], 0)
        self.assertEqual(snap["prefetch_used_ready"] + snap["prefetch_used_waited"] + snap["prefetch_unused"], snap["prefetch_issued"])
        self.assertEqual(snap["io_errors"], 0)
        self.assertEqual(snap["cache_misses"], snap["ram_hits"] + snap["ssd_misses"])
        self.assertEqual(snap["h2d_bytes"], (snap["ram_hits"] + snap["ssd_misses"] + snap["prefetch_issued"]) * s2.object_payload_bytes)


if __name__ == "__main__":
    unittest.main()
