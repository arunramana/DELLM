#!/bin/bash
# Start a Hivemind bootstrap DHT node
echo "Starting Hivemind bootstrap node on port ${DHT_PORT:-12345}..."
python -m hivemind.dht --host_maddrs /ip4/0.0.0.0/tcp/${DHT_PORT:-12345}
