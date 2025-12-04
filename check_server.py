"""Quick script to check if server is running."""
import requests
import sys

try:
    response = requests.get("http://localhost:8000/health", timeout=5)
    if response.status_code == 200:
        data = response.json()
        print("✓ Server is running!")
        print(f"  Status: {data.get('status')}")
        print(f"  Orchestrator initialized: {data.get('orchestrator_initialized')}")
        sys.exit(0)
    else:
        print(f"✗ Server responded with status {response.status_code}")
        sys.exit(1)
except requests.exceptions.ConnectionError:
    print("✗ Server is not running!")
    print("  Start it with: python minimal/main.py")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error checking server: {e}")
    sys.exit(1)

