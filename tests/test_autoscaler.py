import sys

from lmm_serving.common import scheduling
from lmm_serving.common import autoscaling

_passed = _failed = 0


def check(name, cond):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  ok   {name}")
    else:
        _failed += 1
        print(f"  FAIL {name}")


def test_active_set_least_loaded():
    print("[least-loaded active set]")
    s = scheduling.LeastLoadedScheduler(["a", "b", "c", "d"])
    s.set_active(2)
    picks = {s.acquire(1) for _ in range(20)}
    check("only first 2 nodes used", picks == {"a", "b"})
    s.set_active(4)
    picks = {s.acquire(1) for _ in range(40)}
    check("all 4 used after scale up", picks == {"a", "b", "c", "d"})
    s.set_active(99)
    check("clamps to node count", s._active == 4)
    s.set_active(0)
    check("clamps to >=1", s._active == 1)


def test_active_set_round_robin():
    print("[round-robin active set]")
    s = scheduling.RoundRobinScheduler(["a", "b", "c", "d"])
    s.set_active(2)
    seq = [s.acquire() for _ in range(4)]
    check("round-robins within active 2", seq == ["a", "b", "a", "b"])


def test_load_window():
    print("[sliding-window load]")
    w = autoscaling.LoadWindow(window_s=10)
    w.add(100, now=1000)
    w.add(100, now=1002)
    # 200 tokens over a 10s window -> 20 tok/s
    check("rate = sum/window", w.rate_per_s(now=1003) == 20.0)
    # At now=1008 the cutoff is 998, both events still in window.
    check("keeps in-window events", w.rate_per_s(now=1008) == 20.0)
    # At now=1015 the cutoff is 1005, both events (1000, 1002) evicted -> 0.
    check("evicts old events", w.rate_per_s(now=1015) == 0.0)


def test_desired_active():
    print("[desired_active = clamp(ceil(load/M_C), 1, N)]")
    check("zero load -> 1 (stay ready)", autoscaling.desired_active(0, 1000, 8) == 1)
    check("exact multiple", autoscaling.desired_active(2000, 1000, 8) == 2)
    check("rounds up", autoscaling.desired_active(2001, 1000, 8) == 3)
    check("caps at n_max", autoscaling.desired_active(99999, 1000, 8) == 8)
    check("no profile (M_C=0) -> all", autoscaling.desired_active(500, 0, 8) == 8)


class _FakeLogger:
    def info(self, *a, **k):
        pass

    def exception(self, *a, **k):
        pass


def test_autoscaler_decide_and_apply():
    print("[Autoscaler decide + apply]")
    img = scheduling.LeastLoadedScheduler([f"i{n}" for n in range(8)])
    txt = scheduling.LeastLoadedScheduler([f"t{n}" for n in range(4)])
    iw, tw = autoscaling.LoadWindow(60), autoscaling.LoadWindow(60)
    # image M_C=2680 tok/s, text M_C=7600 tok/s (from the profiler run).
    pools = [
        {"name": "image", "scheduler": img, "window": iw, "m_c": 2680, "n_max": 8},
        {"name": "text", "scheduler": txt, "window": tw, "m_c": 7600, "n_max": 4},
    ]
    a = autoscaling.Autoscaler(pools, interval_s=30, logger=_FakeLogger())

    # Idle: everything scales to the floor of 1.
    d = a.decide(now=100)
    check("idle image -> 1", d["image"] == 1)
    check("idle text -> 1", d["text"] == 1)

    # Push image load ~8000 img-tok/s -> ceil(8000/2680)=3 active image instances.
    now = 1000
    for i in range(8):
        iw.add(8000 * 60 / 8, now=now)  # spread so the 60s-window rate ~= 8000/s
    d = a.decide(now=now)
    check("image load -> ceil(8000/2680)=3", d["image"] == 3)
    a.apply(d)
    check("apply set image active=3", img._active == 3)
    check("text stays 1 (no text load)", d["text"] == 1)


def test_autoscaler_tick_writes_csv():
    print("[Autoscaler tick writes metrics csv]")
    import tempfile, os

    img = scheduling.LeastLoadedScheduler(["i0", "i1"])
    txt = scheduling.LeastLoadedScheduler(["t0", "t1"])
    pools = [
        {
            "name": "image",
            "scheduler": img,
            "window": autoscaling.LoadWindow(60),
            "m_c": 2680,
            "n_max": 2,
        },
        {
            "name": "text",
            "scheduler": txt,
            "window": autoscaling.LoadWindow(60),
            "m_c": 7600,
            "n_max": 2,
        },
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "autoscale.csv")
        a = autoscaling.Autoscaler(
            pools, interval_s=30, logger=_FakeLogger(), metrics_path=path
        )
        a._tick()
        a._tick()
        lines = open(path).read().strip().splitlines()
        check("header + 2 rows", len(lines) == 3)
        check(
            "header has active cols",
            "image_active" in lines[0] and "text_active" in lines[0],
        )


def main():
    for t in (
        test_active_set_least_loaded,
        test_active_set_round_robin,
        test_load_window,
        test_desired_active,
        test_autoscaler_decide_and_apply,
        test_autoscaler_tick_writes_csv,
    ):
        t()
    print(f"\n{_passed} passed, {_failed} failed")
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
