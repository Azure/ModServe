import sys

from lmm_serving.launcher import env, launch
from lmm_serving.launcher import topology as topo

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


def _by_name(nodes, name):
    return next(n for n in nodes if n.name == name or n.name.startswith(name))


def _recv(nodes):
    return _by_name(nodes, "transfer-recv")


# ── Single-box (host-independent) ─────────────────────────────────────────────


def test_single_box_modserve():
    print("[single-box local.yaml]")
    t = topo.load("modserve", "local.yaml")
    nodes = launch._modserve_nodes(t, "test")
    recv = _recv(nodes)
    n_images = len(t.image_nodes)

    check("recv rank = image count", recv.env["TRANSFER_RANKS"] == f"{n_images}:0")
    check("world size = images + 1", recv.env["WORLD_SIZE"] == str(n_images + 1))
    check("loopback master", recv.env["MASTER_ADDR"] == "127.0.0.1")
    check("IB disabled on single box", recv.env.get("NCCL_IB_DISABLE") == "1")
    check("socket iface = lo (not eth0)", recv.env["NCCL_SOCKET_IFNAME"] == "lo")
    check("all nodes local", all(env.is_local(n.host) for n in nodes))


# ── Multi-box topology + NCCL wiring (host-independent) ───────────────────────


def test_multi_box_topology():
    print("[3-box internvl_3box.yaml: 4 TP4 text + 8 TP1 image, multi-text-node]")
    t = topo.load("modserve", "internvl_3box.yaml")
    nodes = launch._modserve_nodes(t, "test")
    recvs = [n for n in nodes if n.name.startswith("transfer-recv")]
    texts = [n for n in nodes if n.name.startswith("text ")]
    images = [n for n in nodes if n.name.startswith("image")]
    n_images = len(t.image_nodes)
    recv0 = recvs[0]

    # NCCL world = 8 image (ranks 0..7) + 4 transfer-recv (ranks 8..11) = 12.
    check("8 image nodes", len(images) == 8)
    check("4 text nodes", len(texts) == 4)
    check("4 transfer-recv nodes", len(recvs) == 4)
    check("world size = 12", recv0.env["WORLD_SIZE"] == str(n_images + len(texts)))
    check("master = rank-0 image host (box1)", recv0.env["MASTER_ADDR"] == "10.0.0.4")
    check("socket iface = eth0", recv0.env["NCCL_SOCKET_IFNAME"] == "eth0")
    check("IB HCAs pinned", recv0.env.get("NCCL_IB_HCA", "").startswith("mlx5_ib"))
    check("IB not disabled", "NCCL_IB_DISABLE" not in recv0.env)

    # Each text node owns a distinct recv rank N+j and a distinct recv port 5051+j.
    recv_ranks = sorted(int(r.env["TRANSFER_RANKS"].split(":")[0]) for r in recvs)
    check("recv ranks 8..11", recv_ranks == [8, 9, 10, 11])
    recv_ports = sorted(int(r.env["TENSOR_TRANSFER_PORT"]) for r in recvs)
    check("recv ports 5051..5054", recv_ports == [5051, 5052, 5053, 5054])

    paired = all(
        r.host == x.host
        and r.env["CUDA_VISIBLE_DEVICES"] == x.env["CUDA_VISIBLE_DEVICES"]
        and r.env["TENSOR_TRANSFER_PORT"] == x.env["TENSOR_TRANSFER_RECV_PORT"]
        for r, x in zip(recvs, texts)
    )
    check("each recv paired to its text node (host+gpu+port)", paired)

    # Every image node carries a distinct sender rank 0..7.
    ranks = sorted(int(n.env["IMAGE_NODE_RANK"]) for n in images)
    check("image ranks 0..7", ranks == list(range(8)))


def test_ssh_transport():
    print("[ssh transport]")
    t = topo.load("modserve", "internvl_3box.yaml")
    nodes = launch._modserve_nodes(t, "test")
    img = next(
        n for n in nodes if n.name.startswith("image") and not env.is_local(n.host)
    )
    argv = launch._ssh_argv(img)

    check("uses ssh -T", argv[:2] == ["ssh", "-T"])
    check("targets the node host", img.host in argv)
    remote = argv[-1]
    check("cd to repo root", f"cd {env.REPO_ROOT}" in remote)
    check("exec env (clean handoff)", "exec env" in remote)
    check("forwards IMAGE_NODE_RANK", "IMAGE_NODE_RANK=" in remote)
    check("runs the component", "image_node_manager.py" in remote)
    check("absolute python (no PATH dependence)", str(env.python_exe()) in remote)
    check("forwards NCCL_SOCKET_IFNAME", "NCCL_SOCKET_IFNAME=eth0" in remote)
    check("forwards NCCL_IB_HCA", "NCCL_IB_HCA=" in remote)
    check("forwards MASTER_ADDR", "MASTER_ADDR=10.0.0.4" in remote)
    # A clean PATH (conda bin) is forwarded, not box1's session PATH.
    check("forwards conda bin on PATH", env.bin_dir() in remote)
    check("no box1 session PATH leakage", "vscode" not in remote)


# ── Host locality mechanism (portable) ───────────────────────────────────────


def test_host_locality_mechanism():
    print("[host locality mechanism (portable)]")
    check("loopback is local", env.is_local("127.0.0.1"))
    # TEST-NET-3 (RFC 5737) — never assigned to a real interface, so always remote.
    check("foreign IP is remote", not env.is_local("203.0.113.1"))


# ── Topology validation ───────────────────────────────────────────────────────


def test_validate():
    print("[topology validation]")
    check("local modserve clean", topo.validate(topo.load("modserve")) == [])
    check("local monolith clean", topo.validate(topo.load("monolith")) == [])

    Node, Topology = topo.Node, topo.Topology
    # Image encoders always run TP1 (launch hardcodes it) — reject anything else.
    p = topo.validate(
        Topology(
            "modserve",
            text_nodes=[Node("h:8080", "TP1", [0])],
            image_nodes=[Node("h:8086", "TP2", [], rank=0, gpu_id="1")],
        )
    )
    check("image TP2 rejected", any("always run TP1" in m for m in p))

    # Malformed input must produce problems, never raise.
    cases = {
        "empty modserve flagged": (
            Topology("modserve"),
            "no modserve_text_nodes",
        ),
        "bad TP flagged": (
            Topology("monolith", monolith_nodes=[Node("h:8080", "TPX", [0])]),
            "unsupported TPX",
        ),
        "non-int gpu_id flagged": (
            Topology(
                "modserve",
                text_nodes=[Node("h:8080", "TP1", [0])],
                image_nodes=[Node("h:8086", "TP1", [], rank=0, gpu_id="4,5")],
            ),
            "single GPU id",
        ),
        "address without port flagged": (
            Topology("monolith", monolith_nodes=[Node("h", "TP1", [0])]),
            "host:port",
        ),
    }
    for name, (t, needle) in cases.items():
        try:
            problems = topo.validate(t)
        except Exception as e:  # noqa: BLE001
            check(name, False)
            print(f"       raised {type(e).__name__}: {e}")
            continue
        check(name, any(needle in m for m in problems))


def main():
    for t in (
        test_single_box_modserve,
        test_multi_box_topology,
        test_ssh_transport,
        test_host_locality_mechanism,
        test_validate,
    ):
        t()
    print(f"\n{_passed} passed, {_failed} failed")
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
