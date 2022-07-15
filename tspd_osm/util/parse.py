import hashlib as hl
from typing import Optional

from pandas import DataFrame

from tspd_osm.types import OSMObject, LatLon
from tspd_osm.distance import calc_euclid_distance


def get_digest(value: str) -> str:
    """
    Get digest of value.
    """
    return hl.sha256(value.encode('utf-8')).hexdigest()


def sort_locations(locations: list[list[str, str]]) -> list[list[str, str]]:
    """
    Sort locations.
    """
    return sorted(locations, key=lambda x: (x[0], x[1]))


def parse_osm_objects_to_lac_lons(osm_objects: list[OSMObject]) -> list[LatLon]:
    """
    Parse osm objects to lac lons.
    """
    return [LatLon(lat=float(osm_object.lat), lon=float(osm_object.lon)) for osm_object in osm_objects]


def parse_osm_objects_to_str_lists(osm_objects: list[OSMObject]) -> list[(str, str)]:
    """
    Parse osm objects to lon lat tuples.
    """
    return [[osm_object.lon, osm_object.lat] for osm_object in osm_objects]


def parse_osm_objects_to_euclid_distances(lat_lons: list[LatLon]) -> DataFrame:
    """
    Parse osm objects to euclid distance.
    """
    index = range(len(lat_lons))

    return DataFrame(
        data=[

            [
                calc_euclid_distance(start, end)
                for end in lat_lons
            ]
            for start in lat_lons
        ],
        index=index,
        columns=index,
        dtype=float
    )


def parse_str_to_float(value: str) -> Optional[float]:
    """
    Parse string to float.
    """
    try:
        return float(value)
    except ValueError:
        return None

