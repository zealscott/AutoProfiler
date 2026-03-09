"""Data cleaning utilities for PII deduplication."""


def deduplicate(pii_list: list) -> list:
    """Remove duplicate PII entries by attribute type.

    Keeps the entry with the highest confidence score for each type.

    Args:
        pii_list: List of PII dicts, each with 'type', 'confidence', 'evidence', 'guess'.

    Returns:
        Deduplicated list with one entry per attribute type.
    """
    best = {}
    for entry in pii_list:
        if not isinstance(entry, dict):
            continue
        attr_type = entry.get("type", "")
        if not attr_type:
            continue

        # Parse confidence to float for comparison
        try:
            confidence = float(entry.get("confidence", 0))
        except (ValueError, TypeError):
            confidence = 0.0

        if attr_type not in best or confidence > best[attr_type][1]:
            best[attr_type] = (entry, confidence)

    return [entry for entry, _ in best.values()]
