import json
import logging
import hivemind
from hivemind.utils import get_dht_time

logger = logging.getLogger(__name__)


def create_dht(initial_peers=None, port=None, start=True):
    """Create and return a Hivemind DHT instance."""
    kwargs = {}
    if initial_peers:
        peers = [p.strip() for p in initial_peers if p.strip()]
        if peers:
            kwargs["initial_peers"] = peers
    if port:
        kwargs["host_maddrs"] = [f"/ip4/0.0.0.0/tcp/{port}"]
    kwargs["start"] = start
    return hivemind.DHT(**kwargs)


def store_value(dht, key: str, value: dict, expiration_time: float = 300.0) -> bool:
    """Store a JSON-serializable value in the DHT."""
    json_str = json.dumps(value)
    subkey = dht.peer_id.to_base58() if hasattr(dht, "peer_id") else "default"
    success = dht.store(
        key=key,
        subkey=subkey,
        value=json_str.encode("utf-8"),
        expiration_time=get_dht_time() + expiration_time,
    )
    if success:
        logger.debug(f"Stored key={key}")
    else:
        logger.warning(f"Failed to store key={key}")
    return success


def get_value(dht, key: str) -> dict | None:
    """Retrieve a value from the DHT by key."""
    result = dht.get(key, latest=True)
    if result is None or result.value is None:
        return None
    # result.value is a dictionary of subkey -> (value, expiration)
    if isinstance(result.value, dict):
        # Get most recent entry
        for subkey, (val, _exp) in result.value.items():
            if isinstance(val, bytes):
                return json.loads(val.decode("utf-8"))
            elif isinstance(val, str):
                return json.loads(val)
    return None
