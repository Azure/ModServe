import sys

from lmm_serving.common import scheduling

_passed = 0
_failed = 0


def check(name, cond):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  ok   {name}")
    else:
        _failed += 1
        print(f"  FAIL {name}")


def test_num_pending_tokens():
    print("[modality-aware pending tokens: DecOnly sums text+image, CroAttn text-only]")
    # DecOnly (InternVL) counts text+image; CroAttn (Llama) text-only.
    check(
        "DecOnly image+text = sum",
        scheduling.num_pending_tokens(350, 1024, decoder_only=True) == 1374,
    )
    check(
        "CroAttn ignores image tokens",
        scheduling.num_pending_tokens(350, 4096, decoder_only=False) == 350,
    )
    # Robust to stringy inputs (payloads are JSON).
    check(
        "coerces str inputs",
        scheduling.num_pending_tokens("350", "10", decoder_only=True) == 360,
    )


def test_least_loaded_routing():
    print("[least-loaded text routing]")
    sched = scheduling.make_scheduler("least_loaded", ["A", "B", "C"])
    # First three acquires spread across the (equally-loaded) nodes.
    first = sched.acquire(100)
    second = sched.acquire(100)
    third = sched.acquire(100)
    check("spreads across distinct nodes", {first, second, third} == {"A", "B", "C"})
    heavy = sched.acquire(1000)  # one node -> 1100
    light1 = sched.acquire(1)  # must avoid the 1100 node
    light2 = sched.acquire(1)
    check("avoids the heaviest node", heavy not in (light1, light2))
    # Releasing refunds exactly the charged load.
    before = sched._load[heavy]
    sched.release(heavy, 1000)
    check("release refunds load", sched._load[heavy] == before - 1000)


def test_round_robin_ignores_weight():
    print("[round-robin baseline ignores weight]")
    sched = scheduling.make_scheduler("round_robin", ["A", "B"])
    seq = [sched.acquire(w) for w in (9999, 1, 1, 1)]
    check("strict alternation regardless of weight", seq == ["A", "B", "A", "B"])


def test_scheduler_factory():
    print("[scheduler factory]")
    # The two non-obvious branches: None defaults to round-robin, bad name raises.
    check(
        "default is round_robin",
        isinstance(
            scheduling.make_scheduler(None, ["a"]), scheduling.RoundRobinScheduler
        ),
    )
    try:
        scheduling.make_scheduler("bogus", ["a"])
        check("rejects unknown scheduler", False)
    except ValueError:
        check("rejects unknown scheduler", True)


def main():
    for t in (
        test_num_pending_tokens,
        test_least_loaded_routing,
        test_round_robin_ignores_weight,
        test_scheduler_factory,
    ):
        t()
    print(f"\n{_passed} passed, {_failed} failed")
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
