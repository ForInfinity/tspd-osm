import os
from random import sample
from typing import List

import pandas as pd

from .types import GeoList, MapDataset, OSMObject
from .network import open_route_matrix_query, nominatim_place_query
from tspd_osm.util.parse import parse_osm_objects_to_euclid_distances, parse_osm_objects_to_lac_lons, \
    parse_osm_objects_to_str_lists
from tspd_osm.util.log_util import get_logger

logger = get_logger().getChild('dataset')


def load_geo_list(path: str) -> List[GeoList]:
    """
    Load geo list from path.
    """
    geo_list_data = pd.read_excel(
        path,
        dtype={
            'PostalCode': str,
        },
        na_values=[''],
    )
    geo_list = [
        GeoList(
            city=r['City'],
            number_of_data_points=int(r['NumberOfPoints']),
            places=r['Places'].split(','),
            postal_code=r['PostalCode'],
            country_code=(r['CountryCode'] if r['CountryCode'] and r['CountryCode'].strip() != "" else None),
        )
        for _, r in geo_list_data.iterrows()
    ]

    logger.info(f'Loaded {len(geo_list)} geo lists')

    return geo_list


def load_dataset(path: str) -> MapDataset:
    """
    Load dataset from path.
    """
    return MapDataset.from_xlsx(path)


def generate_map_dataset(places: list[OSMObject], name: str = None) -> MapDataset:
    """
    Generate map dataset from OSMObject list.
    :param name: Name of the dataset. Default is `MapDataset`.
    :type name: str
    :param places: OSMObject list
    :return: MapDataset
    """
    lat_lons = parse_osm_objects_to_lac_lons(places)
    lon_lat_strs = parse_osm_objects_to_str_lists(places)
    df_road_distances = open_route_matrix_query(lon_lat_strs).to_distance_dataframe()
    df_euclid_distances = parse_osm_objects_to_euclid_distances(lat_lons)

    df_meta = pd.DataFrame(
        data=[
            [idx, v.address, v.lat, v.lon] for idx, v in enumerate(places)
        ],
        columns=['id', 'description', 'lat', 'lon'],
    )

    df_meta.set_index('id', inplace=True)
    return MapDataset.from_dataframes(
        dataframes={
            'meta': df_meta,
            'euclid_distance': df_euclid_distances,
            'road_distance': df_road_distances,
        },
        name=name,
    )


def fetch_data_from_geo_list(file_path: str = './geo_list.xlsx'):
    logger.info('Fetching data from geo list..')
    geo_list = load_geo_list(file_path)
    used_names = set()
    for geo in geo_list:
        target_number = geo.number_of_data_points
        dataset_name = f'{geo.country_code}_{geo.city}_{geo.postal_code}_{target_number}'.replace(' ', '_')
        if dataset_name in used_names:
            nr = 1
            new_name = f"{dataset_name}_({nr})"
            while new_name in used_names:
                nr += 1
                new_name = f"{dataset_name}_({nr})"
            dataset_name = new_name
        used_names.add(dataset_name)

        os.makedirs('./data/', exist_ok=True)
        filename = f'./data/{dataset_name}.xlsx'
        if os.path.exists(filename):
            logger.info(f'{filename} already exists, skipping')
            continue
        osm_objects_results: list[object] = []
        extra_osm_objects: list[object] = []
        logger.info(
            f'Fetching {geo.number_of_data_points} data for {geo.city} {geo.postal_code} in {geo.country_code}')
        mean_len = int(target_number / len(geo.places))
        for target in geo.places:

            result = nominatim_place_query(target=target, city=geo.city, postal_code=geo.postal_code,
                                           country_code=geo.country_code, limit_count=target_number)

            result_length = len(result)
            logger.debug(f'\t=> {result_length} results found for: {target}')
            if result and result_length > 0:
                if result_length > mean_len:
                    osm_objects_results.extend(result[:mean_len])
                    extra_osm_objects.extend(result[mean_len:])
                else:
                    osm_objects_results.extend(result)
            else:
                logger.warning(f'No results for {target} in {geo.city}')
        if len(osm_objects_results) < target_number:
            logger.info(f'{len(osm_objects_results)} OSM objects found, extending to {target_number}')
            osm_objects_results.extend(
                sample(extra_osm_objects, min(target_number - len(osm_objects_results), len(extra_osm_objects))))

        if len(osm_objects_results) < target_number / 2:
            logger.error(f'only {len(osm_objects_results)} OSM objects found, aborting..')
            raise Exception(
                f'only {len(osm_objects_results)} OSM objects found, please readjust parameters of {dataset_name}!')
        if len(osm_objects_results) < target_number:
            logger.warning(f'{len(osm_objects_results)} OSM objects found, but 30 expected')
        dataset = generate_map_dataset(list(map(lambda v: OSMObject.from_json(v), iter(osm_objects_results))),
                                       name=dataset_name)
        dataset.write_to_xlsx(filename)
