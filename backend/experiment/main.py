import hivemind

import hivemind
dht = hivemind.DHT(
    host_maddrs=["/ip4/0.0.0.0/tcp/0", "/ip4/0.0.0.0/udp/0/quic"],
    initial_peers=[
    ], start=True)
print("To join the training, use initial_peers =", [str(addr) for addr in dht.get_visible_maddrs()])

print("Successfully connected to the Go DHT server.")
print("My Hivemind peer ID is:", dht.peer_id)

# # You can now see the peers in the network (which should include your Go server)
# import time
# time.sleep(5) # Give it a moment to find peers
# print("Visible peers in the network:", [str(peer) for peer in dht.get_visible_peers()])