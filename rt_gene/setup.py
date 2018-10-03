from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup
import distutils.log
distutils.log.set_verbosity(distutils.log.DEBUG)  # Set DEBUG level

d = generate_distutils_setup(
    packages=['rt_gene'],
    package_dir={'': 'src'}
)

setup(**d)

