def summarize_repairs(records):
    """Return aggregate repair-loop statistics for proposal experiments."""
    total = len(records)
    if total == 0:
        return {"avg_repairs": 0.0}
    return {
        "avg_repairs": sum(record.get("repair_iterations", 0) for record in records) / total
    }
