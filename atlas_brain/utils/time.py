"""Shared time-formatting utilities."""


def format_minutes(minutes: int, *, round_to: int = 0) -> str:
    """Format minutes-since-midnight as '3:45 PM'.

    Args:
        minutes: Minutes since midnight (0–1439).
        round_to: Round to nearest N minutes (0 = no rounding).
    """
    if round_to > 0:
        minutes = round(minutes / round_to) * round_to
    minutes = max(0, min(minutes, 1439))
    h, m = divmod(minutes, 60)
    period = "AM" if h < 12 else "PM"
    display_h = h % 12 or 12
    return f"{display_h}:{m:02d} {period}"
