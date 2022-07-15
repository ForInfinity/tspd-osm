from math import radians, pow, cos, sin, asin, sqrt

import geopy

from .types import LatLon, OSMObject, MapDataset


def calc_euclid_distance(start: LatLon, end: LatLon) -> float:
    """
    Calculate euclid distance between two points in meter.
    :param start: (longitude, latitude)
    :param end: (longitude, latitude)
    :return: distance in meters
    """

    earth_diameter = 6378137.0  # in meters
    lat_diff = end.rad_lat - start.rad_lat
    lon_diff = end.rad_lon - start.rad_lon

    distance = 2 * asin(sqrt(
        pow(sin(lat_diff / 2), 2)
        + cos(start.rad_lat) * cos(end.rad_lat)
        * pow(sin(lon_diff / 2), 2)
    )) * earth_diameter

    return distance



def calc_lat_distance(start: LatLon, end: LatLon) -> float:
    """
    Calculate latitude distance between two points in meter.
    :param start: (longitude, latitude)
    :param end: (longitude, latitude)
    :return: distance in meters
    """

    end = LatLon(end.deg_lat, start.deg_lon)
    return calc_euclid_distance(start, end)


def calc_lon_distance(start: LatLon, end: LatLon) -> float:
    """
    Calculate longitude distance between two points in meter.
    :param start: (longitude, latitude)
    :param end: (longitude, latitude)
    :return: distance in meters
    """
    end = LatLon(start.deg_lat, end.deg_lon)
    return calc_euclid_distance(start, end)
