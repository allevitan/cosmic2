#!/usr/bin/env python
import os
import shutil
import sys
from setuptools import setup, find_packages, Extension
from glob import glob

CURRENT_PYTHON = sys.version_info[:2]
REQUIRED_PYTHON = (3, 0)

if CURRENT_PYTHON < REQUIRED_PYTHON:
    sys.stderr.write("""
==========================
Unsupported Python version
==========================
This version of ptychocam requires Python {}.{}, but you're trying to
install it on Python {}.{}.
""".format(*(REQUIRED_PYTHON + CURRENT_PYTHON)))
    sys.exit(1)


VERSION = '1.0'
#version = __import__('ptychocam').get_version()

short_description = '''Preprocessing module for the COSMIC beamline.'''

# The source dist comes with batteries included, the wheel can use pip to get the rest
is_wheel = 'bdist_wheel' in sys.argv

excluded = []
if is_wheel:
    excluded.append('extlibs.future')

def exclude_package(pkg):
    for exclude in excluded:
        if pkg.startswith(exclude):
            return True
    return False

def create_package_list(base_package):
    return ([base_package] +
            [base_package + '.' + pkg
             for pkg
             in find_packages(base_package)
             if not exclude_package(pkg)])

def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as f:
        return f.read()

udp_ext = Extension('cosmicp.udpframereader',
		
                           sources = ['cosmicp/udpframereader.c'],
		
                           extra_compile_args=['-std=c99','-march=native','-O3'],
		
                           extra_link_args=['-std=c99','-march=native'])

EXCLUDE_FROM_PACKAGES = []

print("Found packages")
print(find_packages())

CONFIG_PATH = os.path.join(os.path.expanduser('~')) + "/cosmicp_config"

# if the config directory tree doesn't exist, create it
if not os.path.exists(CONFIG_PATH):
    os.makedirs(CONFIG_PATH)

shutil.copy('configuration/default.json', CONFIG_PATH)

setup_info = dict(
    name='cosmicp',
    version=VERSION,
    python_requires='>={}.{}'.format(*REQUIRED_PYTHON),
    author='Pablo Enfedaque',
    author_email='pablo.enfedaque@gmail.com',
    url='https://github.com/lbl-camera/cosmic2.git',
    download_url='https://github.com/lbl-camera/cosmic2.git',
    description=short_description,
    license='OSS',
    include_package_data=True,
    entry_points={},

    install_requires=['numpy', 'h5py==2.10.0', 'Pillow', 'scipy', 'tifffile', 'mpi4py', 'pyzmq', 'msgpack', 'msgpack_numpy', 'absl-py'],

    extras_require={
        "cupy": ["cupy"],
        "mpi4py": ["mpi4py"]
    },

    data_files=[('src', glob('cosmicp/*.py'))],

    scripts= glob('cosmicp/cosmic.py'),

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research', 
        'License :: OSI Approved :: OSS License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics'
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],

    project_urls={
        'Source': 'https://github.com/lbl-camera/cosmic2/'
   },

    ext_modules = [udp_ext],


    # Package info
    packages=create_package_list('cosmicp'),

    # Add _ prefix to the names of temporary build dirs
    options={
        'build': {'build_base': '_build'}
    },
    zip_safe=False,
)

setup(**setup_info)

