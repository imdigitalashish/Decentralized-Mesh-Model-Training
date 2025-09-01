package main

import (
	"bytes"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"flag"
	"fmt"
	"log"
	"math/big"
	"net"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/mr-tron/base58"
	"github.com/vmihailenco/msgpack/v5"
)

const (
	K_BUCKET_SIZE = 20
	ALPHA         = 3
	NODE_ID_LEN   = 32
)

const (
	REQ_PING      = 1
	REQ_STORE     = 2
	REQ_FIND_NODE = 3
)

const (
	RES_PONG        = 1
	RES_OK          = 2
	RES_FOUND_NODES = 3
	RES_ERROR       = 5
)

type NodeID [NODE_ID_LEN]byte

func (nid NodeID) String() string {
	return hex.EncodeToString(nid[:])
}

type Peer struct {
	ID   NodeID
	IP   net.IP
	Port int
}

func (p Peer) String() string {
	return fmt.Sprintf("Peer<%s>(%s:%d)", p.ID.String()[:10], p.IP, p.Port)
}

type KBucket struct {
	peers []*Peer
	lock  sync.RWMutex
}

func (kb *KBucket) Add(p *Peer) {
	kb.lock.Lock()
	defer kb.lock.Unlock()
	for i, existingPeer := range kb.peers {
		if bytes.Equal(existingPeer.ID[:], p.ID[:]) {
			kb.peers = append(kb.peers[:i], kb.peers[i+1:]...)
			kb.peers = append([]*Peer{p}, kb.peers...)
			return
		}
	}
	if len(kb.peers) < K_BUCKET_SIZE {
		kb.peers = append([]*Peer{p}, kb.peers...)
		return
	}
	kb.peers = kb.peers[:len(kb.peers)-1]
	kb.peers = append([]*Peer{p}, kb.peers...)
}

func (kb *KBucket) Peers() []*Peer {
	kb.lock.RLock()
	defer kb.lock.RUnlock()
	peersCopy := make([]*Peer, len(kb.peers))
	copy(peersCopy, kb.peers)
	return peersCopy
}

type RoutingTable struct {
	nodeID  NodeID
	buckets [NODE_ID_LEN * 8]*KBucket
	lock    sync.RWMutex
}

func NewRoutingTable(nodeID NodeID) *RoutingTable {
	rt := &RoutingTable{nodeID: nodeID}
	for i := 0; i < NODE_ID_LEN*8; i++ {
		rt.buckets[i] = &KBucket{}
	}
	return rt
}

func (rt *RoutingTable) getBucketIndex(id NodeID) int {
	distance := getDistance(rt.nodeID, id)
	if distance.BitLen() == 0 {
		return 0
	}
	return (NODE_ID_LEN * 8) - 1 - distance.BitLen()
}

func (rt *RoutingTable) Add(p *Peer) {
	if p.ID == rt.nodeID {
		return
	}
	index := rt.getBucketIndex(p.ID)
	bucket := rt.buckets[index]
	bucket.Add(p)
}

func (rt *RoutingTable) FindClosest(targetID NodeID, count int) []*Peer {
	rt.lock.RLock()
	defer rt.lock.RUnlock()
	allPeers := []*Peer{}
	for _, bucket := range rt.buckets {
		allPeers = append(allPeers, bucket.Peers()...)
	}
	sort.Slice(allPeers, func(i, j int) bool {
		distI := getDistance(targetID, allPeers[i].ID)
		distJ := getDistance(targetID, allPeers[j].ID)
		return distI.Cmp(distJ) == -1
	})
	if len(allPeers) > count {
		return allPeers[:count]
	}
	return allPeers
}

type DHT struct {
	NodeID       NodeID
	conn         *net.UDPConn
	routingTable *RoutingTable
	store        sync.Map
	pendingRPCs  sync.Map
}

func nodeIDToPeerID(id NodeID) (string, error) {
	multihash := make([]byte, 2+NODE_ID_LEN)
	multihash[0] = 0x12
	multihash[1] = 0x20
	copy(multihash[2:], id[:])
	return base58.Encode(multihash), nil
}

func NewDHT(listenAddr string) (*DHT, error) {
	addr, err := net.ResolveUDPAddr("udp", listenAddr)
	if err != nil {
		return nil, fmt.Errorf("could not resolve UDP address: %w", err)
	}
	conn, err := net.ListenUDP("udp", addr)
	if err != nil {
		return nil, fmt.Errorf("could not listen on UDP: %w", err)
	}
	log.Printf("DHT Server listening on %s", conn.LocalAddr())
	nodeID := generateNodeID(conn.LocalAddr().String())
	log.Printf("Generated Node ID: %s", nodeID.String())
	peerID, err := nodeIDToPeerID(nodeID)
	if err != nil {
		return nil, fmt.Errorf("could not generate peer id: %w", err)
	}
	log.Printf("Generated libp2p-compatible Peer ID: %s", peerID)
	listenUDPAddr := conn.LocalAddr().(*net.UDPAddr)
	ipStr := listenUDPAddr.IP.String()
	if listenUDPAddr.IP.IsUnspecified() {
		ipStr = "127.0.0.1"
		log.Printf("Listening on unspecified address, use the following multiaddress for local connections:")
	}
	log.Printf("==================================================================================")
	log.Printf("Your Hivemind initial_peer multiaddress is:")
	log.Printf("/ip4/%s/udp/%d/p2p/%s", ipStr, listenUDPAddr.Port, peerID)
	log.Printf("==================================================================================")
	dht := &DHT{
		NodeID:       nodeID,
		conn:         conn,
		routingTable: NewRoutingTable(nodeID),
	}
	return dht, nil
}

func (dht *DHT) Listen() {
	buf := make([]byte, 65535)
	for {
		n, raddr, err := dht.conn.ReadFromUDP(buf)
		if err != nil {
			log.Printf("Error reading from UDP: %v", err)
			continue
		}
		go dht.handlePacket(buf[:n], raddr)
	}
}

func (dht *DHT) Bootstrap(initialPeers []string) error {
	if len(initialPeers) == 0 {
		log.Println("No initial peers provided, starting as a new network.")
		return nil
	}
	var wg sync.WaitGroup
	log.Printf("Bootstrapping with peers: %v", initialPeers)
	for _, peerAddr := range initialPeers {
		wg.Add(1)
		go func(peerAddr string) {
			defer wg.Done()
			hostPortAddr := peerAddr
			if strings.HasPrefix(peerAddr, "/") {
				var err error
				hostPortAddr, err = parseMultiaddr(peerAddr)
				if err != nil {
					log.Printf("Could not parse bootstrap multiaddress %s: %v", peerAddr, err)
					return
				}
			}
			addr, err := net.ResolveUDPAddr("udp", hostPortAddr)
			if err != nil {
				log.Printf("Could not resolve bootstrap peer address %s: %v", hostPortAddr, err)
				return
			}
			log.Printf("Pinging bootstrap peer %s", addr.String())
			resp, err := dht.sendRequest(addr, REQ_PING, dht.NodeID[:])
			if err != nil {
				log.Printf("Failed to ping bootstrap peer %s: %v", addr.String(), err)
				return
			}
			if len(resp) < 3 {
				log.Printf("Invalid PONG response from %s", addr.String())
				return
			}
			peerIDBytes, ok := resp[2].([]byte)
			if !ok || len(peerIDBytes) != NODE_ID_LEN {
				log.Printf("Invalid Node ID in PONG from %s", addr.String())
				return
			}
			var peerID NodeID
			copy(peerID[:], peerIDBytes)
			peer := &Peer{
				ID:   peerID,
				IP:   addr.IP,
				Port: addr.Port,
			}
			dht.routingTable.Add(peer)
			log.Printf("Successfully added bootstrap peer: %s", peer)
		}(peerAddr)
	}
	wg.Wait()
	log.Println("Discovering more nodes near self...")
	dht.findNode(dht.NodeID)
	return nil
}

func (dht *DHT) handlePacket(data []byte, raddr *net.UDPAddr) {
	var msg []interface{}
	if err := msgpack.Unmarshal(data, &msg); err != nil {
		log.Printf("Failed to unmarshal msgpack from %s: %v", raddr, err)
		return
	}
	if len(msg) < 2 {
		log.Printf("Invalid message format: too short from %s", raddr)
		return
	}
	msgType, ok := msg[0].(int64)
	if !ok {
		log.Printf("Invalid message format: type is not an int from %s", raddr)
		return
	}
	rpcID, ok := msg[1].([]byte)
	if !ok {
		log.Printf("Invalid message format: rpc_id is not bytes from %s", raddr)
		return
	}
	if ch, loaded := dht.pendingRPCs.Load(string(rpcID)); loaded {
		select {
		case ch.(chan []interface{}) <- msg:
		default:
		}
		return
	}
	switch msgType {
	case REQ_PING:
		dht.handlePing(msg, raddr)
	case REQ_STORE:
		dht.handleStore(msg, raddr)
	case REQ_FIND_NODE:
		dht.handleFindNode(msg, raddr)
	default:
		log.Printf("Received unknown request type %d from %s", msgType, raddr)
	}
}

func (dht *DHT) handlePing(msg []interface{}, raddr *net.UDPAddr) {
	if len(msg) < 3 {
		return
	}
	rpcID, _ := msg[1].([]byte)
	senderIDBytes, ok := msg[2].([]byte)
	if !ok || len(senderIDBytes) != NODE_ID_LEN {
		log.Printf("Invalid PING request from %s", raddr)
		return
	}
	var senderID NodeID
	copy(senderID[:], senderIDBytes)
	peer := &Peer{ID: senderID, IP: raddr.IP, Port: raddr.Port}
	dht.routingTable.Add(peer)
	log.Printf("Received PING from %s", peer)
	dht.sendResponse(raddr, RES_PONG, rpcID, dht.NodeID[:])
}

func (dht *DHT) handleStore(msg []interface{}, raddr *net.UDPAddr) {
	if len(msg) < 6 {
		return
	}
	rpcID, _ := msg[1].([]byte)
	senderIDBytes, _ := msg[2].([]byte)
	key, keyOK := msg[3].([]byte)
	value, valOK := msg[4].([]byte)
	if !keyOK || !valOK || len(senderIDBytes) != NODE_ID_LEN {
		log.Printf("Invalid STORE request from %s", raddr)
		dht.sendResponse(raddr, RES_ERROR, rpcID, "invalid store request")
		return
	}
	var senderID NodeID
	copy(senderID[:], senderIDBytes)
	peer := &Peer{ID: senderID, IP: raddr.IP, Port: raddr.Port}
	dht.routingTable.Add(peer)
	log.Printf("Storing key '%s' from %s", hex.EncodeToString(key)[:10], peer)
	dht.store.Store(string(key), value)
	dht.sendResponse(raddr, RES_OK, rpcID, true)
}

func (dht *DHT) handleFindNode(msg []interface{}, raddr *net.UDPAddr) {
	if len(msg) < 4 {
		return
	}
	rpcID, _ := msg[1].([]byte)
	senderIDBytes, _ := msg[2].([]byte)
	keyBytes, keyOK := msg[3].([]byte)
	if !keyOK || len(senderIDBytes) != NODE_ID_LEN || len(keyBytes) != NODE_ID_LEN {
		log.Printf("Invalid FIND_NODE request from %s", raddr)
		return
	}
	var senderID, keyID NodeID
	copy(senderID[:], senderIDBytes)
	copy(keyID[:], keyBytes)
	peer := &Peer{ID: senderID, IP: raddr.IP, Port: raddr.Port}
	dht.routingTable.Add(peer)
	log.Printf("Handling FIND_NODE for key %s from %s", keyID.String()[:10], peer)
	closestPeers := dht.routingTable.FindClosest(keyID, K_BUCKET_SIZE)
	var responsePeers []string
	for _, p := range closestPeers {
		ipStr := p.IP.String()
		if ipv4 := p.IP.To4(); ipv4 != nil {
			ipStr = ipv4.String()
		}
		peerIDStr, err := nodeIDToPeerID(p.ID)
		if err != nil {
			log.Printf("Could not convert node ID %s to peer ID: %v", p.ID.String(), err)
			continue
		}
		maddr := fmt.Sprintf("/ip4/%s/udp/%d/p2p/%s", ipStr, p.Port, peerIDStr)
		responsePeers = append(responsePeers, maddr)
	}
	dht.sendResponse(raddr, RES_FOUND_NODES, rpcID, responsePeers)
}

func (dht *DHT) sendResponse(addr *net.UDPAddr, respType int64, rpcID []byte, payload ...interface{}) {
	msg := []interface{}{respType, rpcID}
	msg = append(msg, payload...)
	data, err := msgpack.Marshal(msg)
	if err != nil {
		log.Printf("Failed to marshal response: %v", err)
		return
	}
	_, err = dht.conn.WriteToUDP(data, addr)
	if err != nil {
		log.Printf("Failed to send response to %s: %v", addr, err)
	}
}

func (dht *DHT) sendRequest(addr *net.UDPAddr, reqType int64, payload ...interface{}) ([]interface{}, error) {
	rpcID := make([]byte, 8)
	rand.Read(rpcID)
	msg := []interface{}{reqType, rpcID}
	msg = append(msg, payload...)
	data, err := msgpack.Marshal(msg)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}
	respChan := make(chan []interface{}, 1)
	dht.pendingRPCs.Store(string(rpcID), respChan)
	defer dht.pendingRPCs.Delete(string(rpcID))
	_, err = dht.conn.WriteToUDP(data, addr)
	if err != nil {
		return nil, fmt.Errorf("failed to send request to %s: %w", addr, err)
	}
	select {
	case resp := <-respChan:
		return resp, nil
	case <-time.After(5 * time.Second):
		return nil, fmt.Errorf("request timeout to %s", addr)
	}
}

func (dht *DHT) findNode(targetID NodeID) {
	log.Printf("Looking for nodes close to %s", targetID.String()[:10])
	closestPeers := dht.routingTable.FindClosest(targetID, ALPHA)
	for _, peer := range closestPeers {
		addr := &net.UDPAddr{IP: peer.IP, Port: peer.Port}
		go func(p *Peer, a *net.UDPAddr) {
			resp, err := dht.sendRequest(a, REQ_FIND_NODE, dht.NodeID[:], targetID[:])
			if err != nil {
				log.Printf("FIND_NODE request to %s failed: %v", p, err)
				return
			}
			if len(resp) < 3 {
				return
			}
			foundPeersData, ok := resp[2].([]interface{})
			if !ok {
				return
			}
			for _, item := range foundPeersData {
				peerData, ok := item.([]interface{})
				if !ok || len(peerData) < 3 {
					continue
				}
				idBytes, _ := peerData[0].([]byte)
				ipStr, _ := peerData[1].(string)
				port, _ := peerData[2].(int64)
				if idBytes == nil || len(idBytes) != NODE_ID_LEN || net.ParseIP(ipStr) == nil {
					continue
				}
				var newPeerID NodeID
				copy(newPeerID[:], idBytes)
				newPeer := &Peer{
					ID:   newPeerID,
					IP:   net.ParseIP(ipStr),
					Port: int(port),
				}
				log.Printf("Discovered new peer via FIND_NODE: %s", newPeer)
				dht.routingTable.Add(newPeer)
			}
		}(peer, addr)
	}
}

func parseMultiaddr(maddr string) (string, error) {
	parts := strings.Split(maddr, "/")
	if len(parts) < 5 {
		return "", fmt.Errorf("invalid multiaddr format: %s", maddr)
	}
	if parts[1] != "ip4" || parts[3] != "udp" {
		return "", fmt.Errorf("unsupported multiaddr protocol (only ip4/udp supported): %s", maddr)
	}
	ip := parts[2]
	port := parts[4]
	return net.JoinHostPort(ip, port), nil
}

func generateNodeID(seed string) NodeID {
	hasher := sha256.New()
	hasher.Write([]byte(seed))
	var id NodeID
	copy(id[:], hasher.Sum(nil))
	return id
}

func getDistance(id1, id2 NodeID) *big.Int {
	buf1 := new(big.Int).SetBytes(id1[:])
	buf2 := new(big.Int).SetBytes(id2[:])
	return new(big.Int).Xor(buf1, buf2)
}

func main() {
	listenAddr := flag.String("listen", "0.0.0.0:0", "Address and port to listen on for DHT traffic.")
	initialPeersRaw := flag.String("peers", "", "Comma-separated list of initial peers (e.g., /ip4/127.0.0.1/udp/4000/p2p/Qm...,127.0.0.1:4001).")
	flag.Parse()
	dht, err := NewDHT(*listenAddr)
	if err != nil {
		log.Fatalf("Failed to create DHT node: %v", err)
	}
	var initialPeers []string
	if *initialPeersRaw != "" {
		initialPeers = strings.Split(*initialPeersRaw, ",")
	}
	if len(initialPeers) > 0 {
		err = dht.Bootstrap(initialPeers)
		if err != nil {
			log.Printf("Bootstrap process finished with errors: %v", err)
		}
	}
	dht.Listen()
}
