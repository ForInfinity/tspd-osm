import os
import time
from io import BytesIO
from typing import Optional

import requests

from OSMPythonTools.nominatim import Nominatim
from PIL import Image

from tspd_osm.util.cache import write_cache, read_cache, get_cache_path
from tspd_osm.types import ORSMatrixResponse
from tspd_osm.util.parse import sort_locations

from tspd_osm.util.log_util import get_logger

logger = get_logger().getChild("network")

min_nominatim_query_gap = 1
min_osm_tile_query_gap = 1

nominatim = Nominatim()

_rate_limit_dict: dict[str, float] = {}


def _rate_limit(key: str, min_time_gap: float):
    """
    Limit the number of requests to the Service and wait until min_time_gap reached.

    :param key: the key of the rate limit
    :type key: str
    :param min_time_gap: the minimum time gap between two requests
    :type min_time_gap:  float
    :return: None
    :rtype: None
    """
    global _rate_limit_dict
    last_request_time = _rate_limit_dict.get(key, 0.0)
    current_time = time.time()
    delta = current_time - last_request_time
    if delta < min_time_gap:
        sleep_time = max(0., min_time_gap - delta)
        logger.debug('{}: Rate limited, waiting {} seconds'.format(key, sleep_time))
        time.sleep(sleep_time)
    _refresh_rate_limit(key)


def _refresh_rate_limit(key: str):
    """
    Refresh the rate limit for a given key.

    :param key: the key to refresh
    :type key: str
    :return: None
    :rtype: None
    """
    global _rate_limit_dict
    _rate_limit_dict[key] = time.time()


def rate_limit_nominatim_query(str, params: any):
    """
    Query Nominatim using cache and rate limit.
    :param str: the query string
    :type str: str
    :param params: the query parameters
    :type params: any
    :return: the result of the query
    :rtype: any
    """
    _rate_limit("nominatim", min_nominatim_query_gap)
    return nominatim.query(str, params=params)


def nominatim_place_query(target="school", postal_code="", city="", country_code=None, limit_count=None):
    """
    Query Nominatim using cache.
    :param target: the type of the place to query
    :type target: str
    :param postal_code: the postal code of the place to query
    :type postal_code: str
    :param city:  the city of the place to query
    :type city: str
    :param country_code: the country code of the place to query
    :type country_code: str
    :param limit_count: the maximum number of results to return
    :type limit_count: int
    :return: list[dict]
    :rtype: list[dict]
    """
    query_string = "{} near {} {}".format(target, postal_code, city)
    logger.debug("Nominatim query: {}".format(query_string))
    params = {
        "accept-language": "en-US",
    }
    if country_code:
        params["countrycodes"] = country_code
    if limit_count:
        params["limit"] = max(limit_count, 30)

    result = rate_limit_nominatim_query(query_string, params=params).toJSON()
    if len(result) == 0:
        logger.warning(
            "Nominatim query returned empty result for query: {} with params {}".format(query_string, params))
    return result


__ors_api_url = "https://api.openrouteservice.org/v2/matrix/driving-car"

__min_ors_query_gap = 40 / 60


def open_route_matrix_query(locations: list[[str, str]]) -> ORSMatrixResponse:
    """
    Query Distance Matrix using cache.
    :param locations: a list of [longitude, latitude] **str**
    :type locations: list[[str, str]]
    :return: list[list[float]]
    :rtype:
    """

    cache_name = "ors_matrix"
    cache_key = str(sort_locations(locations))

    # check cache
    result = read_cache(cache_name, cache_key)
    if result is None:
        # cache missed, make a query to server
        logger.debug("Cache missed: {}".format(locations))
        # check api key
        api_key = os.environ['OPEN_ROUTE_SERVICE_API_KEY']
        if api_key is None:
            raise AttributeError("OPEN_ROUTE_SERVICE_API_KEY not set in .env file")

        # throttle
        _rate_limit("open_route_service_query", min_nominatim_query_gap)
        response = requests.post(
            url=__ors_api_url,
            headers={
                'Accept': 'application/json, application/geo+json, application/gpx+xml, img/png; charset=utf-8',
                'Authorization': api_key,
                'Content-Type': 'application/json; charset=utf-8'
            },
            json={
                "locations": locations,
                "metrics": ["distance", "duration"],
                "resolve_locations": "true",
            }
        )
        if response.status_code != 200:
            logger.error("ORS query failed with code {}: {}".format(response.status_code, response.text))
            raise ConnectionError({"status": response.status_code, "message": response.reason})

        # request successful
        result = response.json()

        # save to cache
        write_cache(cache_name, cache_key, result)

        if len(result) == 0:
            logger.warning("ORS query returned empty result for query: {}".format(locations))

        logger.debug("Content cached with key {}".format(cache_key))

    result = ORSMatrixResponse(
        durations=result["durations"],
        distances=result["distances"],
        sources=result["sources"],
        destinations=result["destinations"],
        metadata=result["metadata"],
    )
    # parse result to data frame
    return result


def fetch_image_tile(x_tile, y_tile, zoom) -> Optional[Image.Image]:
    """
    Fetch a tile from OpenStreetMap. See: [1]

    :param x_tile: x tile number
    :type x_tile: int
    :param y_tile:  y tile number
    :type y_tile:  int
    :param zoom:  zoom level
    :type zoom:  int
    :return:  an image object
    :rtype:  Optional[Image.Image]

    [1] https://operations.osmfoundation.org/policies/tiles/
    """
    smurl = r"https://a.tile.openstreetmap.org/{0}/{1}/{2}.png"
    tile: Image = None
    cache_name = get_cache_path("street_img", f"{zoom}_{x_tile}_{y_tile}")
    if os.path.exists(cache_name):
        try:
            tile = Image.open(cache_name)
        except OSError:
            logger.warning("Failed to open image file: {}\n".format(cache_name))
    if tile is None:
        imgurl = smurl.format(zoom, x_tile, y_tile)
        # logger.debug("\x1b[1K\rDownload from URL: " + imgurl)
        _rate_limit("street_img_query", min_osm_tile_query_gap)
        response = requests.get(
            url=imgurl,
            headers={
                'User-Agent': 'tspd-osm/0.1 (Windows NT 10.0; Win64; x64) Python/3.6.5 (A Tsp-Drone Solver)',
                'Cache-Control': 'max-age=2592000'
            }
        )
        tile = Image.open(BytesIO(response.content))
        # logger.debug("\x1b[1K\rSaving to cache: {}".format(cache_name))
        with open(cache_name, "wb") as f:
            f.truncate(0)
            tile.save(f, format="PNG")
    return tile


