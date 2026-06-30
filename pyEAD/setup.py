from setuptools import setup, find_packages

setup(
    name="PYEAD",
    version="0.1.0",
    description="PYthon Energy and Angle Distribution estimator (PYEAD)",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="D.C. Easley",
    author_email="easleydc@ornl.gov",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "joblib",
    ],
    python_requires=">=3.9",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License:",
    ],
    include_package_data=True,
)
