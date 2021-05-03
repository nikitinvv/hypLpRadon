from pkg_resources import get_distribution, DistributionNotFound

from hypRadon.hypRadon import hypRadon
from hypRadon import lpRgpu

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass
