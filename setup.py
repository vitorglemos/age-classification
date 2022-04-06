import os
import sys
import platform
import subprocess
from setuptools import setup, find_packages

path = os.path.join(os.path.dirname(__file__), 'age_classification', '__version__.py')
with open(path, 'r') as f:
    content = f.read()
    __version__ = content[content.find('=') + 1:].strip()[1: -1]


if not os.environ.get('AGE_DEPLOY'):
    print('Installing dependencies...')
    filter_bin = list(filter(lambda x: 'pip' in x, os.listdir(os.path.dirname(sys.executable))))
    filter_bin.sort(key=lambda x: len(x))
    pip_path = os.path.join(os.path.dirname(sys.executable), filter_bin[-1])
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    install_dependencies = subprocess.Popen([pip_path, 'install', '-r', requirements_path],
                                            shell=(platform.system() == "Windows"))
    install_dependencies.wait()
    if install_dependencies.returncode != 0:
        raise Exception('Failed to install dependencies.')

setup(name='age_classification',
      version=__version__,
      url='https://github.com/vitorglemos/age-classification',
      author='VitorLemos',
      include_package_data=True,
      description='Base API for age model',
      packages=find_packages())