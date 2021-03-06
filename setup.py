from setuptools import setup
from setuptools import find_packages
import os

lib_folder = os.path.dirname(os.path.realpath(__file__))
requirement_path = lib_folder + '/requirements.txt'
install_requires = []
if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        install_requires = f.read().splitlines()
setup(
    name='tspd-osm',
    version='0.0.3',
    packages=['tspd_osm'],
    url='',
    license='',
    author='J. Xiao',
    author_email='junfeng.xiao@outlook.com',
    description='',
    install_requires=install_requires
)
