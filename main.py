import logging
import os

from dotenv import load_dotenv

from tspd_osm.dataset import fetch_data_from_geo_list

load_dotenv()


def main():
    logging.basicConfig(level=logging.INFO)
    logging.info('Started')
    fetch_data_from_geo_list(os.environ['GET_LIST_FILE_PATH'])


if __name__ == "__main__":
    main()
