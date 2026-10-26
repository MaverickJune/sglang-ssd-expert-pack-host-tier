# Unit test for the miss_tier_probe instrumentation using the synthetic pack
# fixture from SGLang's own expert-pack runtime tests. Run with:
#   PYTHONPATH=/opt/ep/sglang/test/registered/expert_pack python -m pytest bench/test_miss_tier_probe.py
import ctypes, os, tempfile, unittest
from pathlib import Path
import torch
from test_expert_pack_runtime import _make_pack
from sglang.srt.layers.moe.expert_pack import ExpertPackStore


def _store(pack, manifest, digests, **kw):
    return ExpertPackStore(
        pack, manifest_path=manifest, expected_layers=1, expected_experts=2,
        expected_top_k=1, expected_source_sha256=digests["source"],
        expected_model_identity_sha256=digests["model_identity"],
        expected_config_sha256=digests["config"], cache_vram_mib=1, stage_slots=1, **kw)


class TestMissTierProbe(unittest.TestCase):
    def test_probe_disabled_adds_no_keys(self):
        with tempfile.TemporaryDirectory() as d:
            pack, manifest, digests = _make_pack(Path(d))
            s = _store(pack, manifest, digests)
            self.assertNotIn("miss_pagecache_resident", s.stats)
            self.assertIsNone(s._probe_map)
            s.close()

    def test_residency_absent_then_resident(self):
        with tempfile.TemporaryDirectory() as d:
            pack, manifest, digests = _make_pack(Path(d))
            s = _store(pack, manifest, digests, miss_tier_probe=True)
            self.assertTrue(s.stats["miss_tier_probe"])
            # Evict the pack from the page cache, then expert 1 must read as absent.
            fd = os.open(pack, os.O_RDONLY)
            os.fsync(fd); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
            resident, total = s._probe_residency(0, 1)
            self.assertEqual(total, s.object_payload_bytes)
            self.assertEqual(resident, 0)
            # Buffered read populates the page cache -> fully resident.
            staging = torch.empty(s.object_payload_bytes, dtype=torch.uint8)
            s._read_object_range(0, 1, staging, start=0, length=s.object_payload_bytes)
            resident, total = s._probe_residency(0, 1)
            self.assertEqual(resident, total)
            s._record_miss_tier(0, total, 5_000_000)      # 5 ms -> absent bucket
            s._record_miss_tier(total, total, 50_000)      # 50 us -> resident bucket
            self.assertEqual(s.stats["miss_pagecache_absent"], 1)
            self.assertEqual(s.stats["miss_pagecache_resident"], 1)
            self.assertEqual(s.stats["miss_pagecache_absent_ns"], 5_000_000)
            self.assertEqual(s.stats["read_latency_hist_us"], {"8192": 1, "64": 1})
            snap = s.snapshot()
            self.assertIn("read_latency_hist_us", snap)
            s.close()
            self.assertIsNone(s._probe_map)


if __name__ == "__main__":
    unittest.main()
