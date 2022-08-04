from distutils.core import setup
import subprocess

PYSUFFIX = subprocess.run(["python3-config", "--extension-suffix"], capture_output=True, text=True).stdout.strip("\n")
# PYSUFFIX = '.cpython-38-x86_64-linux-gnu.so'

setup(
    name='localpsfcpp',
    version='0.1dev',
    packages=['localpsfcpp',],
    license='MIT',
    author='Nick Alger',
    # long_description=open('README.txt').read(),
    include_package_data=True,
    package_data={'localpsfcpp': ['localpsfcpp'+PYSUFFIX]},
)