import hashlib
import json


def content_hash(record: dict) -> str:
    """Hash semantic content only (messages + tradition + lang), not metadata.

    Per VALIDATION_REPORT.md M5: hash must be deterministic over content fields
    only, so that re-generated records with different timestamps still deduplicate
    correctly against existing corpus entries.
    """
    content = {
        "messages": record["messages"],
        "tradition": record["tradition"],
        "lang": record["lang"],
    }
    canonical = json.dumps(content, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
