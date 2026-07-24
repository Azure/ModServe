import sys

from lmm_serving.profiler import model as rfmodel

_passed = _failed = 0


def check(name, cond):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  ok   {name}")
    else:
        _failed += 1
        print(f"  FAIL {name}")


def _synthetic_points():
    pts = []
    for tp in (1, 2, 4):
        for batch in (1, 2, 4, 8):
            for unit in (256, 1024, 4096):
                lat = 20.0 + (batch * unit) / (tp * 50.0)
                pts.append(
                    {
                        "parallelism": tp,
                        "batch": batch,
                        "load_unit": unit,
                        "latency_ms": lat,
                    }
                )
    return pts


def test_rf_max_capacity():
    print("[RF model: M_C off the fitted surface]")
    sm = rfmodel.StageModel.fit(_synthetic_points(), stage="test")
    batch_grid = [1, 2, 4, 8]
    load_grid = list(range(256, 4097, 256))
    tight = sm.max_capacity(2, slo_ms=100, batch_grid=batch_grid, load_grid=load_grid)
    relaxed = sm.max_capacity(
        2, slo_ms=2000, batch_grid=batch_grid, load_grid=load_grid
    )
    check("relaxed SLO >= tight SLO M_C", relaxed["m_c_per_s"] >= tight["m_c_per_s"])
    impossible = sm.max_capacity(
        2, slo_ms=1, batch_grid=batch_grid, load_grid=load_grid
    )
    check("M_C=0 when SLO unmeetable", impossible["m_c_per_s"] == 0.0)


def main():
    for t in (test_rf_max_capacity,):
        t()
    print(f"\n{_passed} passed, {_failed} failed")
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
