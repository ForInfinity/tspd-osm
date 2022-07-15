from typing import Optional, Literal, Dict, Any

from pandas import DataFrame

import pandas as pd

from math import radians, degrees
from tspd_osm.util.log_util import get_logger

logger = get_logger().getChild('types')


class LatLon:
    """
    A class that represents a latitude and longitude.
    """
    def __init__(self, lat: float, lon: float, unit: Literal['radians', 'degrees'] = 'degrees'):
        """
        :param lat: 纬度
        :type lat: float
        :param lon: 经度
        :type lon: float
        """
        if unit == 'degrees':
            self.__deg_lat = lat
            self.__deg_lon = lon
        else:
            self.__deg_lat = degrees(lat)
            self.__deg_lon = degrees(lon)

    @property
    def deg_lat(self):
        """
        latitude in degrees
        """
        return self.__deg_lat

    @property
    def deg_lon(self):
        """
        longitude in degrees
        """
        return self.__deg_lon

    @property
    def rad_lat(self):
        """
        latitude in radians
        """
        return radians(self.deg_lat)

    @property
    def rad_lon(self):
        """
        longitude in radians
        """
        return radians(self.deg_lon)

    def __str__(self):
        return "(deg_lat={}, deg_lon={})".format(self.deg_lat, self.deg_lon)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.deg_lat == other.lat and self.deg_lon == other.lon

    def __hash__(self):
        return hash((self.deg_lat, self.deg_lon))


class GeoList:
    """
    A class that represents a list of geographic place groups.
    """
    def __init__(self, city: str, number_of_data_points: int, places: list[str], postal_code: Optional[str] = None,
                 country_code: Optional[str] = None):
        self.city = city
        if str(postal_code) == 'nan':
            self.postal_code = ''
        else:
            self.postal_code = postal_code
        self.number_of_data_points = number_of_data_points
        self.places = [p.strip() for p in places]
        self.country_code = country_code

    def __str__(self):
        return "GeoList(city={}, postal_code={}, number_of_data_points={}, places={})".format(
            self.city, self.postal_code, self.number_of_data_points, self.places
        )


class MapDataset:
    """
    An map dataset, containing meta data and distance matrices.
    """

    @classmethod
    def __calc_real_index(cls, i: int, j: int) -> (int, int):
        """
        Calculate the real index of the distance matrix.

        :param i: index of the first node
        :type i: int
        :param j: index of the second node
        :type j: int
        :return: the real index of the distance matrix
        :rtype: (int, int)
        """
        if i == j:
            raise ValueError("i and j must not be equal")
        if i < j:
            smaller = i
            bigger = j
        else:
            smaller = j
            bigger = i
        return smaller, bigger - smaller - 1

    @classmethod
    def __read_single_distance_matrix(cls, count: int, dataframe: DataFrame) -> list[list[float]]:
        """
        Read a single distance matrix from a dataframe.

        :param count: the number of nodes
        :type count: int
        :param dataframe: the dataframe
        :type dataframe: DataFrame
        :return: a trimmed distance matrix with no unnecessary 0 values
        :rtype: list[list[float]]
        """

        result: list[list[float]] = []

        for (index, row) in dataframe.iterrows():
            # plus 1 to trim "self distance", plus 1 to trim index
            start = index + 1
            # Ignore the last empty row
            if start >= count:
                break
            row_trimmed: list[float] = [row[i] for i in range(start, count)]
            result.append(row_trimmed)

        return result

    def __init__(self, meta: dict[int, dict[str, any]], euclid_distances: list[list[float]],
                 road_distances: list[list[float]],
                 name: str = None):
        self.__name = name or "MapDataset"
        self.__meta = meta
        # logger.info("self.__meta: {}".format(meta))
        self.__count = len(meta)
        self.__euclid_distance = euclid_distances
        self.__road_distance = road_distances

    @classmethod
    def from_dataframes(cls, dataframes: dict[str, DataFrame], name: str = None):
        """
        Create a MapDataset from a dictionary of dataframes. The dictionary must contain the following keys:
            - `meta`: meta data of the map dataset
            - `euclid_distances`: euclidean distance matrix
            - `road_distances`: road distance matrix

        :param dataframes: A dictionary of dataframes.
        :type dataframes: dict[str, DataFrame]
        :param name: The name of the map dataset.
        :type name: str
        :return: A MapDataset
        :rtype: MapDataset
        """
        # Read metadata
        if dataframes['meta'] is None:
            raise ValueError("MapDataset must have a meta dataframe")
        name = name or "MapDataset"
        meta: dict[int, dict[str, any]] = dict()
        for index, row in dataframes['meta'].iterrows():
            # logger.debug("Read meta data: {}".format(row))
            meta[index] = {
                'description': row.get('description'),
                'lat': row.get('lat', 'nan'),
                'lon': row.get('lon', 'nan'),
            }
        count: int = len(meta)
        # logger.debug("meta: {}".format(meta))
        # Read distance matrix
        euclid_distance: list[list[float]] = \
            cls.__read_single_distance_matrix(count, dataframes['euclid_distance'])
        road_distance: list[list[float]] = \
            cls.__read_single_distance_matrix(count, dataframes['road_distance'])
        return cls(meta, euclid_distance, road_distance, name)

    def get_euclid_distance(self, i: int, j: int) -> float:
        """
        Get the euclidean distance between two nodes.

        :param i: index of the first node
        :type i: int
        :param j: index of the second node
        :type j: int
        :return: the euclidean distance between the two nodes
        :rtype: float
        """
        if i == j:
            logger.warning("i and j must not be equal: i={}, j={}".format(i, j))
            return 0.0
        i, j = self.__calc_real_index(i, j)
        return self.__euclid_distance[i][j]

    def get_road_distance(self, i: int, j: int) -> float:
        """
        Get the road distance between two nodes.

        :param i: index of the first node
        :type i: int
        :param j: index of the second node
        :type j: int
        :return: the euclidean distance between the two nodes
        :rtype: float
        """
        if i == j:
            logger.warning("i and j must not be equal: i={}, j={}".format(i, j))
            return 0.0
        i, j = self.__calc_real_index(i, j)
        return self.__road_distance[i][j]

    def calc_road_distance(self, node_ids: list[int]) -> float:
        """
        Calculate the road distance between n nodes.

        **Careful**: The distance between last and first node is included.

        :param node_ids: the node ids
        :type node_ids: list[int]
        :return: the road distance between the nodes
        :rtype: float
        """
        node_count = len(node_ids)
        return sum([self.get_road_distance(node_ids[i], node_ids[(i + 1) % node_count]) for i in range(node_count)])

    @classmethod
    def __to_table_rows(cls, data: list[list[float]]) -> list[list[float]]:
        """
        Convert a distance matrix to a table rows.

        :param data: the distance matrix
        :type data: list[list[float]]
        :return: the table rows
        :rtype: list[list[float]]
        """
        result: list[list[float]] = []

        for row_idx, row in enumerate(data):
            suppliment = []
            for col_idx in range(row_idx):
                i, j = cls.__calc_real_index(col_idx, row_idx)
                suppliment.append(data[i][j])
            arr = suppliment + [0.] + [x for x in row]
            result.append(arr)
        last_row = []
        last_row_idx = len(data)
        for col_idx in range(last_row_idx):
            i, j = cls.__calc_real_index(col_idx, last_row_idx)
            last_row.append(data[i][j])
        result.append(last_row + [0.])
        return result

    def to_array(self, distance_type: Literal["euclid", "road"] = "euclid") -> list[list[float]]:
        """
        Export the distance matrix to an array.

        :param distance_type: the type of distance matrix to export.
        :type distance_type: Literal["euclid", "road"]
        :return: the distance matrix in array
        :rtype: list[list[float]]
        """
        if distance_type == "euclid":
            return self.__to_table_rows(self.__euclid_distance)
        elif distance_type == "road":
            return self.__to_table_rows(self.__road_distance)
        else:
            raise ValueError("Unknown distance type: {}".format(distance_type))

    def write_to_xlsx(self, filename: str):
        """
        Write the distance matrix to an xlsx file.

        :param filename: the filename of the xlsx file
        :type filename: str
        :return: None
        :rtype: None
        """
        df_index = range(self.__count)
        df_cols = [str(i) for i in df_index]

        df_euclid_distance = pd.DataFrame(
            data=self.__to_table_rows(self.__euclid_distance),
            index=df_index,
            columns=df_cols
        )
        df_road_distance = pd.DataFrame(
            data=self.__to_table_rows(self.__road_distance),
            index=df_index,
            columns=df_cols
        )

        df_meta = pd.DataFrame(
            data=[[idx] + list(self.__meta.get(idx).values()) for idx in self.__meta.keys()],
            columns=['id', 'description', 'lat', 'lon']
        )
        logger.info(f"{self.__name}: Write {len(self.__meta)} items to {filename}")
        writer = pd.ExcelWriter(filename)
        df_euclid_distance.to_excel(writer, sheet_name='euclid_distance')
        df_road_distance.to_excel(writer, sheet_name='road_distance')
        df_meta.to_excel(writer, sheet_name='meta')
        writer.save()

    @property
    def euclid_distance_list(self) -> list[list[float]]:
        """
        Get the raw euclidean distance matrix.

        **Careful**: 0 Values are trimmed.

        :return:
        :rtype:
        """
        return self.__euclid_distance

    @property
    def road_distance_list(self) -> list[list[float]]:
        """
        Get the raw road distance matrix.

        **Careful**: 0 Values are trimmed.

        :return:
        :rtype:
        """
        return self.__road_distance

    @classmethod
    def from_xlsx(cls, path: str):
        """
        Load a distance matrix from an xlsx file.

        :param path: the path of the xlsx file
        :type path: str
        :return: the distance matrix
        :rtype: DistanceMatrix
        """
        dataframes: dict[str, DataFrame] = pd.read_excel(
            path,
            sheet_name=None,
            index_col=0,
        )
        return cls.from_dataframes(dataframes)

    @property
    def name(self) -> str:
        """
        Get the name of the distance matrix.
        :return: the name of the distance matrix
        :rtype: str
        """
        return self.__name

    @property
    def meta(self) -> dict[int, dict[str, Any]]:
        """
        Get the meta data of the distance matrix.
        :return: the meta data of the distance matrix
        :rtype: dict[int, dict[str, Any]]
        """
        return self.__meta

    @property
    def count(self) -> int:
        """
        Get the count of nodes.

        :return: the count of the distance matrix
        :rtype: int
        """
        return self.__count

    def __str__(self):
        return "MapDataset(name={}, meta={}, count={})".format(self.__name, self.__meta, self.__count) + \
               "\nEuclid distance:\n{}\nRoad distance:\n{}".format(self.__euclid_distance, self.__road_distance)


class OSMObject:
    """
    A class to represent an OpenStreetMap object, which is returned by the OSM API.
    """
    def __init__(self, osm_id: int, osm_type: str, lat: str, lon: str, address: str):
        self.__osm_id = osm_id
        self.__osm_type = osm_type
        self.__lat = lat
        self.__lon = lon
        self.__address = address
        logger.debug("OSMObject(osm_id={}, osm_type={}, lat={}, lon={}, address={})".format(
            osm_id, osm_type, lat, lon, address))

    @classmethod
    def from_json(cls, json_obj: dict):
        """
        Create an OSMObject from a json object.
        :param json_obj: the json object
        :type json_obj: dict
        :return: the OSMObject
        :rtype: OSMObject
        """
        return cls(
            osm_id=json_obj['osm_id'],
            osm_type=json_obj['osm_type'],
            lat=json_obj['lat'],
            lon=json_obj['lon'],
            address=json_obj['display_name']
        )

    @property
    def osm_id(self) -> int:
        return self.__osm_id

    @property
    def osm_type(self) -> str:
        return self.__osm_type

    @property
    def address(self) -> str:
        return self.__address

    @property
    def lat(self) -> str:
        return self.__lat

    @property
    def lon(self) -> str:
        return self.__lon

    def __str__(self):
        return "OSMObject(osm_id={}, osm_type={}, lat={}, lon={}, address={})".format(
            self.__osm_id, self.__osm_type, self.__lat, self.__lon, self.__address
        )


class ORSMatrixResponse:
    def __init__(self, durations: list[list[float]], distances: list[list[float]], sources: dict, destinations: dict,
                 metadata: dict):
        self.__durations = durations
        self.__distances = distances
        self.__source = sources
        self.__target = destinations
        self.__metadata = metadata

    @classmethod
    def __to_dataframe(cls, data: list[list[float]]):
        df_index = range(len(data))
        return pd.DataFrame(
            data=data,
            index=df_index,
            columns=df_index
        )

    def to_distance_dataframe(self):
        return self.__to_dataframe(self.__distances)

    def to_duration_dataframe(self):
        return self.__to_dataframe(self.__durations)

    @property
    def durations(self) -> list[list[float]]:
        return self.__durations

    @property
    def distances(self) -> list[list[float]]:
        return self.__distances

    @property
    def source(self) -> dict:
        return self.__source

    @property
    def target(self) -> dict:
        return self.__target

    @property
    def metadata(self) -> dict:
        return self.__metadata

    def __str__(self):
        return "ORSMatrixResponse(" \
               "\n\tdurations={}, " \
               "\n\tdistances={}, " \
               "\n\tsource={}, " \
               "\n\ttarget={}, " \
               "\n\tmetadata={}\n)" \
            .format(self.__durations, self.__distances, self.__source, self.__target, self.__metadata)
