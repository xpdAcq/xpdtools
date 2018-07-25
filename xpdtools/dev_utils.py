"""Utilities for making the developer's lives easier"""
import datetime


def _timestampstr(timestamp):
    """Convert timestamp to strftime formate """
    timestring = datetime.datetime.fromtimestamp(float(timestamp)).strftime(
        "%Y%m%d-%H%M%S"
    )
    return timestring
