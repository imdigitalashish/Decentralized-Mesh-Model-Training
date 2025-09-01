import hivemind

import hivemind
dht = hivemind.DHT(
    host_maddrs=["/ip4/0.0.0.0/tcp/0", "/ip4/0.0.0.0/udp/0/quic"],
    initial_peers=[
        "/ip4/10.184.17.85/tcp/55541/p2p/12D3KooWHDi82ZzdiwoPLpzKyFx4AkKQVZ8YzF5AAANACotwLm8n"
    ], start=True)
print("To join the training, use initial_peers =", [str(addr) for addr in dht.get_visible_maddrs()])

print("Successfully connected to the Go DHT server.")
print("My Hivemind peer ID is:", dht.peer_id)
