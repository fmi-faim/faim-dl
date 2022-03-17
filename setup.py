import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="faim-dl",
    version="0.1.0",
    author="Tim-Oliver Buchholz",
    author_email="tim-oliver.buchholz@fmi.ch",
    description="A collection of deep learning models for bio-image analysis.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fmi-faim/faim-dl",
    packages=setuptools.find_packages(),
    python_requires=">=3.9",
    project_urls={
        "Bug Tracker": "https://github.com/fmi-faim/faim-dl/issues"
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache 2.0 License",
        "Operating System :: Linux"
    ],
    install_requires=[
        "torch==1.11",
        "deepspeed==0.5.10",
        "wandb==0.12.11",
        "monai==0.8.1",
        "scikit-learn==1.0.2",
        "mosaicml==0.5",
        "zarr==2.11.1",
        "ome-zarr==0.3.1",
        "pandas==1.4.1"
    ]
)