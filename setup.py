from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='drlkit',
    version='0.0.1',
    description='A High Level Python Deep Reinforcement Learning library. Great for beginners, for prototyping and quickly comparing algorithms',
    py_modules=['drlkit', 'pydrl'],
    package_dir={'':'src'},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)"
        ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FranckNdame/drlkit",
    author="Franck Ndame",
    author_email="franck.mpouli@gmail.com",
)
