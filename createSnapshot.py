from qdrant_client import QdrantClient
import os
import requests

COLLECTION_NAME = "questions"
SNAPSHOT_DIR = "snapshots"

client = QdrantClient(host="localhost", port=6333)

# Create snapshots folder if it doesn't exist
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# Create snapshot
snapshot_info = client.create_snapshot(collection_name=COLLECTION_NAME)
snapshot_name = snapshot_info.name
print(f"[DONE] Snapshot created: {snapshot_name}")

# Download snapshot using requests
url = f"http://localhost:6333/collections/{COLLECTION_NAME}/snapshots/{snapshot_name}"
snapshot_path = os.path.join(SNAPSHOT_DIR, snapshot_name)

response = requests.get(url, stream=True)
if response.status_code == 200:
    with open(snapshot_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"[DOWNLOADED] Snapshot saved at {snapshot_path}")
else:
    print(f"[ERROR] Failed to download snapshot: {response.status_code} {response.text}")
