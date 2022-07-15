from tspd_osm.util.log_util import setup_logging, get_logger

setup_logging()

get_logger().info("""
******************** tspd-osm ********************
   Created by Junfeng Xiao and Ruiyang Zhang

>\tA program that collects sample data from OSM to test
 our TSP-D Solver.
 
CONTRIBUTION:
>\tMaps and Images: © OpenStreetMap contributors (https://openstreetmap.org/copyright)
>\tDistance and Time Matrix: © openrouteservice.org by HeiGIT | Map data © OpenStreetMap contributors'
**************************************************
""")
