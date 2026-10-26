# SPDX-License-Identifier: Apache-2.0
"""Three-tier (VRAM / RAM / SSD) expert placement policy for the Expert Pack runtime.

Pure-Python bookkeeping, deliberately free of torch so it can be unit-tested
without a GPU or a model.  The store (expert_pack.py) owns the buffers, CUDA
events and file reads and asks this object only *where* an expert is and
*which* slot to use.  Policy follows SSD-LLaMA (arXiv 2609.18110) §3.3/§4.2:

* lookup order VRAM -> RAM -> SSD;
* one entry = one complete (layer, expert) gate+up+down block;
* every entry records ``access_count`` and ``last_access`` (a monotonic
  forward-step counter; an access is "selected by the router in a forward
  step", de-duplicated per acquire() update exactly like the existing
  ``cache_hits``/``cache_misses`` counters);
* VRAM admission ("paper"): a newly delivered expert is *retained* only if it
  has been accessed at least ``admission_min_accesses`` times and its previous
  access was at most ``admission_window_steps`` forward steps ago, or if VRAM
  still has a never-used slot; otherwise it executes from a *transient* slot,
  which is the first eviction candidate, so a single cold access cannot evict
  a frequently reused expert.  A transient entry that is hit again and now
  meets the rule is promoted to retained;
* VRAM eviction ("paper"): free slot, else oldest transient slot, else the
  retained entry with the lexicographically smallest (access_count,
  last_access, slot_index); the current update's working set is never
  evicted.  Retained candidates are kept in a lazy min-heap keyed exactly on
  that triple (stale entries are skipped by a per-slot version), so the
  victim is identical to a full scan but costs O(log n) instead of O(n);
* transient pool: ``transient_slots`` extra VRAM slots (allocated by the
  store on top of the retained budget) hold transient entries and are reused
  FIFO (oldest install first), so a slot just read by layer k's kernel is not
  the one layer k+1's H2D overwrites.  When the pool is exhausted within one
  update (large prefill), further transient installs fall back to evicting
  from the retained region exactly as before.  The evicted expert is demoted
  to RAM when RAM has space (free slot, or its bytes are still RAM-resident),
  otherwise discarded;
* RAM eviction: LRU, never an entry whose SSD read / H2D / D2H is in flight
  (the store tells us which slots are busy);
* ``policy="lru"`` reproduces the original reuse-LFU/LRU victim rule.
"""

from __future__ import annotations

import heapq
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Callable, Optional

Key = tuple[int, int]
VRAM, RAM, SSD = "vram", "ram", "ssd"


@dataclass
class VramEntry:
    key: Optional[Key] = None
    retained: bool = True
    last_use_step: int = -1
    seq: int = 0  # install sequence number (FIFO order of transient entries)


@dataclass
class History:
    access_count: int = 0
    last_access: int = -1
    prev_access: int = -1


@dataclass
class TierStats:
    vram_hits: int = 0
    ram_hits: int = 0
    ssd_misses: int = 0
    vram_evictions: int = 0
    demotions_to_ram: int = 0
    demotions_resident: int = 0
    demotions_d2h: int = 0
    demotions_discarded: int = 0
    ram_evictions: int = 0
    transient_uses: int = 0
    promotions: int = 0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class TierPolicy:
    def __init__(
        self,
        vram_slots: int,
        ram_slots: int,
        *,
        policy: str = "paper",
        admission_min_accesses: int = 2,
        admission_window_steps: int = 8,
        transient_slots: int = 0,
    ) -> None:
        if policy not in ("lru", "paper"):
            raise ValueError(f"unknown vram_admission_policy {policy!r}")
        if vram_slots <= 0 or ram_slots < 0 or transient_slots < 0:
            raise ValueError("vram_slots must be positive; ram/transient slots non-negative")
        if admission_min_accesses < 1 or admission_window_steps < 0:
            raise ValueError("invalid admission rule parameters")
        if transient_slots and policy != "paper":
            raise ValueError("transient_slots requires vram_admission_policy='paper'")
        self.policy = policy
        self.admission_min_accesses = admission_min_accesses
        self.admission_window_steps = admission_window_steps
        # Slots [0, retained_slots) are the retained budget (cache_vram_mib);
        # slots [retained_slots, retained_slots + transient_slots) are the pool.
        self.retained_slots = vram_slots
        self.transient_slots = transient_slots
        self.vram: list[VramEntry] = [VramEntry() for _ in range(vram_slots + transient_slots)]
        self._version: list[int] = [0] * len(self.vram)
        self._heap: list[tuple[int, int, int, int]] = []  # (access_count, last_access, idx, version)
        self._fifo: deque[tuple[int, int]] = deque()  # (seq, idx) of transient installs
        self._seq = 0
        self.key_to_vram: dict[Key, int] = {}
        self.vram_lru: "OrderedDict[Key, None]" = OrderedDict()
        self.history: dict[Key, History] = {}
        self.ram_slots = ram_slots
        self.ram_free: list[int] = list(range(ram_slots - 1, -1, -1))
        self.ram_lru: "OrderedDict[Key, int]" = OrderedDict()  # key -> ram slot, LRU order
        self.ram_slot_key: dict[int, Key] = {}
        self.step = 0
        self.stats = TierStats()

    # ------------------------------------------------------------------ steps
    def begin_step(self) -> None:
        """Called once per forward step (the first active MoE layer)."""
        self.step += 1

    def record_access(self, key: Key) -> History:
        h = self.history.setdefault(key, History())
        h.access_count = min(h.access_count + 1, 1 << 30)
        h.prev_access, h.last_access = h.last_access, self.step
        return h

    # ----------------------------------------------------------------- lookup
    def classify(self, key: Key) -> str:
        """Resolve VRAM first, then RAM, then SSD; count the hit/miss tier."""
        idx = self.key_to_vram.get(key)
        if idx is not None:
            self.stats.vram_hits += 1
            self.vram_lru.move_to_end(key)
            entry = self.vram[idx]
            entry.last_use_step = self.step
            if entry.retained:
                self._heap_push(idx)  # score changed (access_count / last_access)
            # A transient entry is never promoted in place: it executes from its
            # temporary slot and is retained on its *next* load if its history
            # then satisfies should_retain().
            return VRAM
        if key in self.ram_lru:
            self.stats.ram_hits += 1
            self.ram_lru.move_to_end(key)
            return RAM
        self.stats.ssd_misses += 1
        return SSD

    def vram_slot_of(self, key: Key) -> int:
        return self.key_to_vram[key]

    def ram_slot_of(self, key: Key) -> Optional[int]:
        return self.ram_lru.get(key)

    # -------------------------------------------------------------- admission
    def has_free_vram(self) -> bool:
        return any(e.key is None for e in self.vram[: self.retained_slots])

    # ----------------------------------------------------------------- heap
    def _heap_push(self, idx: int) -> None:
        e = self.vram[idx]
        h = self.history[e.key]
        self._version[idx] += 1
        heapq.heappush(self._heap, (h.access_count, h.last_access, idx, self._version[idx]))
        if len(self._heap) > 4 * len(self.vram) + 64:
            self._heap = [
                (self.history[v.key].access_count, self.history[v.key].last_access, i, self._version[i])
                for i, v in enumerate(self.vram)
                if v.key is not None and v.retained
            ]
            heapq.heapify(self._heap)

    def _heap_min_retained(self, protected: set) -> Optional[int]:
        """Lowest (access_count, last_access, idx) retained entry not protected."""
        deferred = []
        found = None
        while self._heap:
            count, last, idx, version = self._heap[0]
            e = self.vram[idx]
            if version != self._version[idx] or e.key is None or not e.retained:
                heapq.heappop(self._heap)  # stale
                continue
            if e.key in protected:
                deferred.append(heapq.heappop(self._heap))
                continue
            found = idx
            break
        for item in deferred:
            heapq.heappush(self._heap, item)
        return found

    def _fifo_oldest_transient(self, protected: set, *, pool_only: bool) -> Optional[int]:
        deferred = []
        found = None
        while self._fifo:
            seq, idx = self._fifo[0]
            e = self.vram[idx]
            if e.key is None or e.retained or e.seq != seq:
                self._fifo.popleft()  # stale
                continue
            if e.key in protected or (pool_only and idx < self.retained_slots):
                deferred.append(self._fifo.popleft())
                continue
            found = idx
            break
        self._fifo.extendleft(reversed(deferred))
        return found

    def should_retain(self, key: Key) -> bool:
        if self.policy == "lru":
            return True
        if self.has_free_vram():
            return True
        h = self.history.get(key)
        if h is None or h.access_count < self.admission_min_accesses:
            return False
        return h.prev_access >= 0 and (self.step - h.prev_access) <= self.admission_window_steps

    def frequency(self, key: Key) -> int:
        h = self.history.get(key)
        return min(h.access_count, 255) if h else 0

    def pick_vram_victim(
        self, protected: set, *, retained: bool = True, preserve_oldest: bool = False
    ) -> int:
        """Slot for a new entry that will be installed with ``retained``."""
        if self.policy == "paper" and not retained and self.transient_slots:
            # transient install: free pool slot, else oldest pool entry (FIFO)
            for idx in range(self.retained_slots, len(self.vram)):
                if self.vram[idx].key is None:
                    return idx
            idx = self._fifo_oldest_transient(protected, pool_only=True)
            if idx is not None:
                return idx
            # pool exhausted inside this update: fall through to the retained region
        for idx in range(self.retained_slots):
            if self.vram[idx].key is None:
                return idx
        if self.policy == "lru":
            # Original reuse-LFU-then-LRU rule (expert_pack.py _victim_slot).
            keys = reversed(self.vram_lru) if preserve_oldest else iter(self.vram_lru)
            best, best_freq = None, None
            for key in keys:
                if key in protected:
                    continue
                freq = self.frequency(key)
                if best_freq is None or freq < best_freq:
                    best, best_freq = self.key_to_vram[key], freq
            if best is None:
                raise RuntimeError("expert cache cannot evict the active top-k working set")
            return best
        # paper: transient entries in the retained region first (oldest install),
        # then the retained entry with the lowest (access_count, last_access, idx)
        idx = self._fifo_oldest_transient(protected, pool_only=False)
        if idx is not None and idx < self.retained_slots:
            return idx
        idx = self._heap_min_retained(protected)
        if idx is None:
            raise RuntimeError("expert cache cannot evict the active top-k working set")
        return idx

    def evict_vram(self, idx: int, *, can_d2h: bool) -> tuple[Optional[Key], str]:
        """Remove the VRAM entry in ``idx``.

        Returns (old_key, outcome) with outcome one of:
        ``"none"`` (slot was free), ``"resident"`` (bytes still in RAM -> demotion
        is free, RAM LRU touched), ``"d2h"`` (a free RAM slot was reserved for the
        store to copy the block device-to-host; ``ram_slot_of(old_key)`` gives it),
        ``"discard"`` (no RAM space).
        """
        e = self.vram[idx]
        old = e.key
        if old is None:
            return None, "none"
        self.stats.vram_evictions += 1
        self.key_to_vram.pop(old, None)
        self.vram_lru.pop(old, None)
        self._version[idx] += 1  # invalidates any heap entry for this slot
        e.key, e.retained, e.last_use_step = None, True, -1
        if old in self.ram_lru:
            self.ram_lru.move_to_end(old)
            self.stats.demotions_to_ram += 1
            self.stats.demotions_resident += 1
            return old, "resident"
        if can_d2h and self.ram_free:
            slot = self.ram_free.pop()
            self.ram_lru[old] = slot
            self.ram_slot_key[slot] = old
            self.stats.demotions_to_ram += 1
            self.stats.demotions_d2h += 1
            return old, "d2h"
        self.stats.demotions_discarded += 1
        return old, "discard"

    def install_vram(self, idx: int, key: Key, *, retained: bool) -> None:
        e = self.vram[idx]
        if e.key is not None:
            raise RuntimeError("install into an occupied VRAM slot; evict first")
        self._seq += 1
        e.key, e.retained, e.last_use_step, e.seq = key, retained, self.step, self._seq
        self.key_to_vram[key] = idx
        self.vram_lru[key] = None
        if retained:
            self._heap_push(idx)
        else:
            self.stats.transient_uses += 1
            self._fifo.append((self._seq, idx))

    # -------------------------------------------------------------------- RAM
    def ram_alloc(
        self, key: Key, evictable: Callable[[int], bool]
    ) -> tuple[Optional[int], Optional[Key]]:
        """Reserve a RAM slot for ``key``: a free one, else the LRU entry whose
        slot ``evictable(slot)`` (no in-flight read/H2D/D2H).  Returns
        (slot, evicted_key) or (None, None) when everything is busy
        (the caller applies backpressure and retries)."""
        if key in self.ram_lru:
            self.ram_lru.move_to_end(key)
            return self.ram_lru[key], None
        if self.ram_free:
            slot = self.ram_free.pop()
            self.ram_lru[key] = slot
            self.ram_slot_key[slot] = key
            return slot, None
        for victim_key, slot in self.ram_lru.items():
            if evictable(slot):
                del self.ram_lru[victim_key]
                self.stats.ram_evictions += 1
                self.ram_lru[key] = slot
                self.ram_slot_key[slot] = key
                return slot, victim_key
        return None, None

    def ram_oldest(self) -> Optional[int]:
        for _, slot in self.ram_lru.items():
            return slot
        return None

    def ram_touch(self, key: Key) -> None:
        if key in self.ram_lru:
            self.ram_lru.move_to_end(key)

    def ram_bytes_used(self, payload_bytes: int) -> int:
        return len(self.ram_lru) * payload_bytes

    def ram_resident_keys(self) -> set:
        return set(self.ram_lru)

    # --------------------------------------------------------------- invariant
    def check_invariants(self) -> None:
        s = self.stats
        assert len(self.key_to_vram) == sum(e.key is not None for e in self.vram)
        if self.policy == "paper":
            live = {(self.history[e.key].access_count, self.history[e.key].last_access, i)
                    for i, e in enumerate(self.vram) if e.key is not None and e.retained}
            heap_live = {(c, l, i) for c, l, i, v in self._heap if v == self._version[i]
                         and self.vram[i].key is not None and self.vram[i].retained}
            assert live == heap_live, "heap out of sync with retained entries"
        assert set(self.ram_lru.values()).isdisjoint(self.ram_free)
        assert len(self.ram_lru) + len(self.ram_free) == self.ram_slots
        assert s.demotions_to_ram == s.demotions_resident + s.demotions_d2h
