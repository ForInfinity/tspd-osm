import json
import os
from typing import Optional

from tspd_osm.util.log_util import get_logger
from tspd_osm.util.parse import get_digest

logger = get_logger().getChild('cache')


def get_cache_path(bucket_name: str, key: str) -> str:
    """
    Get the path of the cache file.

    :param bucket_name: the name of the service
    :type bucket_name: str
    :param key: a string that represents the query parameters or object. This key will be digested.
    :type key: str
    :return: the path of the cache file.
    :rtype: str
    """
    filename = "{}-{}".format(bucket_name, get_digest(key))
    rel_path = "cache"
    path = os.path.join(os.getcwd(), rel_path)
    os.makedirs(path,exist_ok=True)
    return os.path.join(path, filename)


def write_cache(name: str, key: str, value: object):
    """
    Write the value to the cache.

    :param name: the name of the service
    :type name: str
    :param key: a string that represents the query parameters. This key will be digested.
    :type key: str
    :param value: the result of query.
    :type value: object
    :return:
    :rtype:
    """
    path = get_cache_path(name, key)
    logger.info("Writing cache: {}".format(path))
    if os.path.exists(path):
        logger.warning("overwriting existing cache {}".format(path))
    f = open(path, 'w')
    try:
        f.write(json.dumps(value))
    except Exception as e:
        logger.error("Failed to write cache: {}".format(e))
    finally:
        f.close()


def read_cache(name: str, key: str) -> Optional[object]:
    """
    Read the value from the cache.

    :param name: the name of the service
    :type name: str
    :param key:  a string that represents the query parameters. This key will be digested.
    :type key: str
    :return: the result of query.
    :rtype: Optional[object]
    """
    path = get_cache_path(name, key)
    if not os.path.exists(path):
        return None
    f = open(path, 'r')
    try:
        obj_str = f.read()
        return json.loads(obj_str)
    finally:
        return None
