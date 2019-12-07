from setuptools import setup
from setuptools import find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='drlkit',
    version='0.0.8',
    description='A High Level Python Deep Reinforcement Learning library. Great for beginners, for prototyping and quickly comparing algorithms',
    install_requires=[
            'torch',
            'numpy',
            'tqdm',
            'matplotlib',
            'box2d-py',
            'gym',
            'keras',
        ],
    license='MIT',
    py_modules=['drlkit'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)",
        "Operating System :: OS Independent",
        ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FranckNdame/drlkit",
    author="Franck Ndame",
    author_email="franck.mpouli@gmail.com",
    packages=find_packages(),
    python_requires='>=3.6',
)
