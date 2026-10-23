import hashlib
import uuid
from datetime import datetime, timezone

from config.settings import EMBEDDING_MODEL


def attach_lineage_metadata(chunks: list, embedding_model: str = EMBEDDING_MODEL) -> None:
    ingested_at = datetime.now(timezone.utc).isoformat()
    for chunk in chunks:
        chunk.metadata.update({
            "chunk_id": str(uuid.uuid4()),
            "chunk_hash": hashlib.sha256(
                chunk.page_content.encode("utf-8")
            ).hexdigest(),
            "ingested_at": ingested_at,
            "embedding_model": embedding_model,
            "modified_at": None,
        })
