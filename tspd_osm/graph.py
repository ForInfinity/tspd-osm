import math
import sys
from typing import Optional, Union, Iterable

import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image
import numpy as np
from matplotlib.figure import Figure

from tspd_osm.distance import calc_euclid_distance, calc_lon_distance, calc_lat_distance
from .network import fetch_image_tile
from tspd_osm.util.parse import parse_str_to_float
from .types import LatLon, MapDataset
from .util.log_util import get_logger
from PIL import ImageFont, ImageDraw

logger = get_logger().getChild("graph")


def deg_to_tile_num(lat_deg, lon_deg, zoom, accurate=False):
    """
    Convert latitude and longitude to OpenStreetMap's tile number.
    
    :param lat_deg: Latitude in degrees.
    :type lat_deg: float
    :param lon_deg: Longitude in degrees.
    :type lon_deg: float
    :param zoom: Zoom level of the image. See: Zoom Levels[1]
    :type zoom: int
    :param accurate: If true, the tile number is returned as float, containing accurate crop information. 
    :type accurate: bool
    :return: Tile number in x and y direction.
    :rtype: int or float
    """
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    x_tile = ((lon_deg + 180.0) / 360.0 * n)
    y_tile = ((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)

    if accurate:
        return x_tile, y_tile
    else:
        return int(x_tile), int(y_tile)


def calculate_tile_count(start: LatLon, end: LatLon, zoom: int) -> int:
    """
    Calculate the number of tiles needed to cover the given latitude and longitude range.
    
    :param start: The vertex with minimal Latitude and minimal Longitude.
    :type start: LatLon
    :param end: The vertex with maximal Latitude and maximal Longitude.
    :type end: LatLon
    :param zoom: Zoom level of the image. See: Zoom Levels[1]
    :type zoom: int
    :return: The number of tiles.
    :rtype: int
    
    [1] https://wiki.openstreetmap.org/wiki/Zoom_levels
    """
    xmin, ymin = deg_to_tile_num(start.deg_lat, start.deg_lon, zoom)
    xmax, ymax = deg_to_tile_num(end.deg_lat, end.deg_lon, zoom)
    count = (xmax - xmin + 1) * (ymax - ymin + 1)
    return count


def get_street_image(start: LatLon, end: LatLon, zoom: int = 12) -> Optional[Image.Image]:
    """
    Get street image of the given latitude and longitude range.

    :param start: Left and Top Vertex of the rectangle.
    :type start: LatLon
    :param end: Right and Bottom Vertex of the rectangle
    :type end: LatLon
    :param zoom: Zoom level of the image. See: Zoom Levels[1]
    :type zoom: int
    :return: The image. If the image could not be downloaded, None is returned.
    :rtype: Optional[Image.Image]
    
    [1] https://wiki.openstreetmap.org/wiki/Zoom_levels
    """
    xmin, ymin = deg_to_tile_num(start.deg_lat, start.deg_lon, zoom, accurate=True)
    xmax, ymax = deg_to_tile_num(end.deg_lat, end.deg_lon, zoom, accurate=True)

    crop_left = xmin % 1 * 256 - 1
    crop_top = ymin % 1 * 256 - 1
    crop_right = (1 - xmax % 1) * 256 - 1
    crop_bottom = (1 - ymax % 1) * 256 - 1

    xmin = int(xmin)
    ymin = int(ymin)
    xmax = int(xmax)
    ymax = int(ymax)

    logger.debug(f"Fetching tile zoom={zoom} start={start} end={end}")

    tile_count_x = xmax - xmin + 1
    tile_count_y = ymax - ymin + 1

    cluster_width = tile_count_x * 256 - 1
    cluster_height = tile_count_y * 256 - 1

    cluster = Image.new('RGB', (cluster_width, cluster_height))
    tile_count = tile_count_x * tile_count_y
    current_tile = 0
    logger.debug(f"Total tiles: {tile_count}")
    for tile_x in range(xmin, xmax + 1):
        for tile_y in range(ymin, ymax + 1):
            try:
                # Read from cache
                current_tile += 1
                print(
                    f"\x1b[1K\r...({current_tile: <3d}/{tile_count: >3d}) Tile: {zoom}_{tile_x}_{tile_y}",
                    end="",
                    flush=True
                )
                tile: Image.Image = fetch_image_tile(tile_x, tile_y, zoom)
                cluster.paste(tile, box=((tile_x - xmin) * 256, (tile_y - ymin) * 255))
            except Exception as e:
                logger.error(f"Couldn't download image tile ({tile_x}, {tile_y})")
                logger.error(e)
                return None
    # crop
    cropped_cluster = cluster.crop((crop_left, crop_top, cluster_width - crop_right, cluster_height - crop_bottom))

    # add Watermark
    font = ImageFont.truetype("arial.ttf", size=15)
    draw = ImageDraw.Draw(cropped_cluster, "RGBA")
    copyright_text = "© OpenStreetMap Contributors"

    draw.rectangle(
        xy=(cropped_cluster.width - 230, cropped_cluster.height - 25, cropped_cluster.width, cropped_cluster.height),
        fill=(254, 254, 254, 127),
    )
    draw.text((cropped_cluster.width - 5, cropped_cluster.height - 5),
              copyright_text,
              font=font,
              fill=(0, 0, 0),
              align="right",
              anchor="rb",
              )

    return cropped_cluster


def get_dataset_street_image(dataset: MapDataset, margin: float = .2) -> tuple[LatLon, LatLon, Image.Image]:
    """
    Get street image of the given dataset.

    :param dataset: The dataset to get the image from.
    :type dataset: MapDataset
    :param margin: The margin to add to the end image. 0 means no margin, 1 means the image is 100% larger in width and height.
    :type margin: float
    :return: The left and top vertex, the right and bottom vertex of the image and the image.
    :rtype: tuple[LatLon, LatLon, Image.Image]
    """
    min_float = float("-inf")
    max_float = float("inf")

    lat_min: float = max_float
    lat_max: float = min_float
    lon_min: float = max_float
    lon_max: float = min_float
    node_metas = dataset.meta.items()
    for node_id, node in node_metas:
        lat = parse_str_to_float(node.get("lat"))
        lon = parse_str_to_float(node.get("lon"))
        if not lat or not lon:
            continue
        if lat < lat_min:
            lat_min = lat
        if lat > lat_max:
            lat_max = lat
        if lon < lon_min:
            lon_min = lon
        if lon > lon_max:
            lon_max = lon

    start = LatLon(lat_max, lon_min)
    end = LatLon(lat_min, lon_max)

    # scale start and end point
    diff_lat = (end.deg_lat - start.deg_lat) * margin / 2.
    diff_lon = (end.deg_lon - start.deg_lon) * margin / 2.

    start = LatLon(start.deg_lat - diff_lat, start.deg_lon - diff_lon)
    end = LatLon(end.deg_lat + diff_lat, end.deg_lon + diff_lon)

    # Calculate zoom factor
    zoom: int = 0
    tile_count = 1
    while tile_count < 8 and zoom < 18:
        zoom += 1
        tile_count = calculate_tile_count(start, end, zoom)
    if tile_count > 20:
        zoom -= 1

    img = get_street_image(start=start, end=end, zoom=zoom)

    if img is None:
        raise Exception("Couldn't download street image")

    return start, end, img


class GraphFactory:
    """
    Factory for creating graphs from given Dataset.
    """

    def __init__(self, dataset: MapDataset, margin: float = .3):
        """
        Create a graph from the given dataset.
        
        :param dataset: The dataset to create the graph from.
        :type dataset: MapDataset
        :param margin: The margin to add to the end image. 0 means no margin, 1 means the map has an extra margin of
         full width and height.
        :type margin: float
        """
        self.__meta = dataset.meta.copy()
        start, end, img = get_dataset_street_image(dataset, margin)
        self.__boundary = (start, end)
        self.__meter_boundary = (calc_lon_distance(start, end), calc_lat_distance(start, end))
        self.__background_image = np.flipud(img)

        # Sizing
        img_size = (img.size[0], img.size[1])
        self.__size = img_size

        self.__base_size = min(self.__size) / 150
        self.__node_size = self.__base_size * 20
        self.__font_size = self.__base_size + 2
        self.__edge_width = self.__base_size / 5

        # Graph
        self.__graph = nx.DiGraph(
            bg_color=[1, 1, 1, 1],
            alpha=1.,
        )

        # Node Positions
        self.__node_positions = {}
        for node_id, node in self.__meta.items():
            lat = parse_str_to_float(node.get("lat"))
            lon = parse_str_to_float(node.get("lon"))
            if not lat or not lon:
                continue
            node_pos = self.parse_coordinate(LatLon(lat, lon))
            self.__node_positions[node_id] = node_pos

        plt.margins(0.0, tight=True)
        plt.rcParams.update({
            "axes.facecolor": (0., 1.0, 0.0, 0.0),  # green with alpha = 50%
        })

        self.clean_axes()
        self.draw_nodes()
        # self.node_graph()

        # Set plot style
        plt.xlim(0, img_size[0])
        plt.ylim(0, img_size[1])
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.tight_layout(pad=0.)
        plt.axis('off')

    def parse_coordinate(self, original: LatLon) -> list[float]:
        """
        Parse a geo-coordinate to the map coordinate system.
        
        :param original: The original coordinate.
        :type original: LatLon
        :return: The parsed coordinate as list[x,y].
        :rtype: list[float]
        """

        delta_x = calc_lon_distance(original, self.__boundary[0])
        delta_y = calc_lat_distance(original, self.__boundary[1])

        scaled_x = delta_x / self.__meter_boundary[0] * self.__size[0]
        # Upside down
        scaled_y = delta_y / self.__meter_boundary[1] * self.__size[1]

        return [scaled_x, scaled_y]

    @property
    def image_size(self) -> tuple[int, int]:
        """
        Get the size of the image.
        
        :return: The size of the image.
        :rtype: tuple[int, int]
        """
        return self.__size

    @property
    def boundary(self) -> tuple[LatLon, LatLon]:
        """
        Get the geo-boundary of the graph.
        
        :return: The geo-boundary of the graph.
        :rtype: tuple[LatLon, LatLon]
        """
        return self.__boundary

    def draw_nodes(self):
        """
        Draw the nodes of the dataset in the graph.
        """

        for n in self.__graph.nodes:
            self.__graph.remove_node(n)

        for node_id, node_pos in self.__node_positions.items():
            self.__graph.add_node(node_id, pos=node_pos)
        # nodes
        nx.draw_networkx_nodes(self.__graph, self.__node_positions, node_size=self.__node_size)

        # labels
        nx.draw_networkx_labels(self.__graph, self.__node_positions,
                                font_size=self.__font_size,
                                font_family='sans-serif',
                                font_color="white",
                                font_weight="bold",
                                verticalalignment="center_baseline")

    def __draw_edges(self, edge_label_dict: dict[tuple[int, int], Optional[str]], edge_args: dict = None,
                     label_args: dict = None):
        """
        Draw the edges of the graph.
        
        :param edge_label_dict: The edges to draw, as a dictionary of (node_id, node_id) -> label or None.
        :type edge_label_dict: dict[tuple[int, int], Optional[str]]
        :param edge_args: The arguments to pass to nx.draw_networkx_edges.
        :type edge_args: dict
        :param label_args: The arguments to pass to nx.draw_networkx_edge_labels.
        :type label_args: dict
        :return: None
        :rtype: None
        """
        edge_args = edge_args or {}
        label_args = label_args or {}

        edges = edge_label_dict.keys()
        labels = {e: edge_label_dict[e] for e in edge_label_dict if edge_label_dict[e] is not None}

        nx.draw_networkx_edges(
            self.__graph,
            self.__node_positions,
            edgelist=edges,
            width=self.__edge_width,
            arrows=True,
            arrowsize=self.__edge_width * 8,
            arrowstyle="-|>",
            **edge_args)

        nx.draw_networkx_edge_labels(self.__graph,
                                     self.__node_positions,
                                     font_size=self.__font_size,
                                     edge_labels=labels,
                                     **label_args)

    def draw_drone_edges(self, edges: dict[tuple[int, int], Optional[str]]):
        """
        Draw the edges of the drone.
        
        :param edges: The edges to draw, as a dictionary of (node_id, node_id) -> label or None.
        :type edges: dict[tuple[int, int], Optional[str]]
        :return: None
        :rtype: None
        """
        self.__draw_edges(edges,
                          edge_args={
                              "edge_color": "blue",
                              "style": "dashed",
                          })

    def draw_truck_edges(self, edges: dict[tuple[int, int], Optional[str]]):
        """
        Draw the edges of the truck.
        
        :param edges: The edges to draw, as a dictionary of (node_id, node_id) -> label or None.
        :type edges: dict[tuple[int, int], Optional[str]]
        :return: None
        :rtype: None
        """
        self.__draw_edges(edges,
                          edge_args={
                          })

    def clean_axes(self):
        """
        Clean all the axes of pyplot.
        """
        axes: list[plt.Axes] = plt.gcf().axes
        for axe in axes:
            axe.remove()
        plt.imshow(self.__background_image, alpha=.5)
        fig: Figure = plt.gcf()
        fig.set_size_inches(self.__size[0] / 100, self.__size[1] / 100)

    def show(self):
        """
        Show the graph.

        :return: None
        :rtype: None
        """
        plt.show()

    def save(self, filename: str):
        """
        Save the graph to a file.

        :param filename:  The filename to save the graph to.
        :type filename:  str
        :return:  None
        :rtype:  None
        """
        plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0)


def __main__():
    dataset = MapDataset.from_xlsx("data/de_Aachen_52066_test.xlsx")

    fac = GraphFactory(dataset)
    fac.draw_drone_edges({
        (0, 1): "D1",
        (1, 2): "D2",
    })
    fac.draw_truck_edges({
        (0, 2): "T1",
    })
    fac.save("test.png")
    fac.show()


if __name__ == '__main__':
    __main__()
