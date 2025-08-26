# -*- coding: utf-8 -*-
"""
Caching functions for the MagpiEM Dash application.
"""

import plotly.graph_objects as go

from .plotting_helpers import (
    create_lattice_plot_from_raw_data,
    create_particle_plot_from_raw_data,
)
from .read_write import load_single_tomogram_raw_data

# Global cache configuration
MAX_CACHE_SIZE = 5
PRELOAD_COUNT = 2  # Number of tomograms to pre-load ahead

# Global pre-loaded tomogram cache organized by session
__preloaded_tomograms = {}  # session_key -> {tomogram_name: figure}


def _get_or_create_cache_entry(
    tomogram_name: str,
    session_key: str,
    cone_size: float = -1,
    show_removed: bool = False,
):
    """
    Get or create a cache entry for a tomogram.

    Parameters
    ----------
    tomogram_name : str
        Name of the tomogram
    session_key : str
        Unique session identifier for this user
    cone_size : float
        Cone size for plotting
    show_removed : bool
        Whether to show removed particles

    Returns
    -------
    tuple
        (session_cache, cached_item) where cached_item is None if not in cache
    """
    global MAX_CACHE_SIZE, __preloaded_tomograms

    if session_key not in __preloaded_tomograms:
        __preloaded_tomograms[session_key] = {}

    session_cache = __preloaded_tomograms[session_key]

    # Use only tomogram name as cache key
    cache_key = tomogram_name
    print(f"DEBUG: Looking for cache key: {cache_key}")

    if cache_key in session_cache:
        cached_item = session_cache.pop(cache_key)
        session_cache[cache_key] = cached_item
        return session_cache, cached_item

    return session_cache, None


def _add_to_cache_and_evict(session_cache: dict, cache_key: str, item, max_size: int):
    """
    Add an item to cache and implement LRU eviction.

    Parameters
    ----------
    session_cache : dict
        The session cache dictionary
    cache_key : str
        Cache key for the item
    item
        Item to cache
    max_size : int
        Maximum cache size
    """
    session_cache[cache_key] = item
    print(f"DEBUG: Added to cache with key: {cache_key}")

    if len(session_cache) > max_size:
        oldest_key = next(iter(session_cache))
        session_cache.pop(oldest_key)
        print(f"DEBUG: Evicted cache entry: {oldest_key}")


def get_cached_tomogram_data(
    tomogram_name: str, data_path: str, session_key: str
) -> list | None:
    """
    Get tomogram data from cache or load it if not cached.
    Implements LRU cache to limit memory usage.

    Parameters
    ----------
    tomogram_name : str
        Name of the tomogram to load
    data_path : str
        Path to the data file
    session_key : str
        Unique session identifier for this user

    Returns
    -------
    list | None
        Tomogram data or None if loading failed
    """
    session_cache, cached_data = _get_or_create_cache_entry(
        tomogram_name, session_key, -1, False
    )

    if cached_data is not None:
        return cached_data

    data = load_single_tomogram_raw_data(data_path, tomogram_name)
    if data is None:
        return None

    _add_to_cache_and_evict(session_cache, tomogram_name, data, MAX_CACHE_SIZE)

    return data


def get_cached_tomogram_figure(
    tomogram_name: str,
    data_path: str,
    session_key: str,
    lattice_data: dict,
    cone_size: float = -1,
    show_removed: bool = False,
) -> go.Figure | None:
    """
    Get tomogram figure from cache or create it if not cached.
    Implements LRU cache to limit memory usage.

    Parameters
    ----------
    tomogram_name : str
        Name of the tomogram to load
    data_path : str
        Path to the data file
    session_key : str
        Unique session identifier for this user
    lattice_data : dict
        Lattice data for this tomogram
    cone_size : float, optional
        Cone size for plotting
    show_removed : bool, optional
        Whether to show removed particles

    Returns
    -------
    go.Figure | None
        Tomogram figure or None if loading failed
    """
    print(f"DEBUG: get_cached_tomogram_figure called for {tomogram_name}")
    print(f"DEBUG: cone_size={cone_size}, show_removed={show_removed}")

    session_cache, cached_figure = _get_or_create_cache_entry(
        tomogram_name, session_key, cone_size, show_removed
    )

    print(f"DEBUG: Cache hit: {cached_figure is not None}")

    if cached_figure is not None:
        # Add lattice IDs to trace text for click detection
        for trace in cached_figure.data:
            if trace.name and trace.name.startswith("Lattice "):
                try:
                    lattice_id = int(trace.name.split(" ")[1])
                    if hasattr(trace, "x") and trace.x is not None:
                        trace.text = [lattice_id] * len(trace.x)
                except (ValueError, IndexError, AttributeError):
                    pass
        print("DEBUG: Returning cached figure")
        return cached_figure

    # Cached figure does not exist, needs to be created
    print("DEBUG: Creating new figure")
    raw_data = load_single_tomogram_raw_data(data_path, tomogram_name)
    if raw_data is None:
        print("DEBUG: Failed to load raw data")
        return None

    # Make lattice plot if possible, otherwise simple plot
    if not lattice_data or tomogram_name not in lattice_data:
        figure = create_particle_plot_from_raw_data(
            raw_data,
            cone_size=cone_size,
        )
    else:
        figure = create_lattice_plot_from_raw_data(
            raw_data,
            lattice_data[tomogram_name],
            cone_size=cone_size,
            show_removed_particles=show_removed,
            selected_lattices=None,
        )

    print("DEBUG: Created new figure, adding to cache")
    cache_key = tomogram_name
    _add_to_cache_and_evict(session_cache, cache_key, figure, MAX_CACHE_SIZE)

    return figure


def preload_tomograms(
    current_tomo_name: str,
    tomogram_names: list,
    data_path: str,
    session_key: str,
    lattice_data: dict,
    cone_size: float = -1,
    show_removed: bool = False,
) -> None:
    """
    Pre-load the current and next few tomogram figures that the user is likely to view.
    This runs in the background to improve user experience.

    Parameters
    ----------
    current_tomo_name : str
        Name of the currently displayed tomogram
    tomogram_names : list
        List of all available tomogram names
    data_path : str
        Path to the data file
    session_key : str
        Unique session identifier for this user
    lattice_data : dict
        Lattice data for all tomograms
    cone_size : float, optional
        Cone size for plotting
    show_removed : bool, optional
        Whether to show removed particles
    """
    global PRELOAD_COUNT

    if not tomogram_names or current_tomo_name not in tomogram_names:
        return

    current_index = tomogram_names.index(current_tomo_name)

    # Preload current tomogram and next few
    for i in range(0, PRELOAD_COUNT + 1):
        next_index = (current_index + i) % len(tomogram_names)
        next_tomo_name = tomogram_names[next_index]

        # Skip preloading if no lattice data exists for this tomogram
        if next_tomo_name not in lattice_data:
            print(f"DEBUG: Skipping preload for {next_tomo_name} - no lattice data")
            continue

        session_cache, cached_figure = _get_or_create_cache_entry(
            next_tomo_name, session_key, cone_size, show_removed
        )
        if cached_figure is None:
            try:
                figure = get_cached_tomogram_figure(
                    next_tomo_name,
                    data_path,
                    session_key,
                    lattice_data,
                    cone_size,
                    show_removed,
                )

            except Exception as e:
                print(f"Pre-loading failed for {next_tomo_name}: {e}")
                continue


def clear_cache(session_key: str):
    """Clear all cached figures for a session."""
    if session_key in __preloaded_tomograms:
        __preloaded_tomograms[session_key].clear()
        print(f"Cleared cache for session {session_key}")


def get_cache_functions():
    """Get a dictionary of cache functions for use in callbacks."""
    return {
        "get_cached_tomogram_figure": get_cached_tomogram_figure,
        "preload_tomograms": preload_tomograms,
        "clear_cache": clear_cache,
    }
