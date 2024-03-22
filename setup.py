from setuptools import setup, find_packages


setup(
    name="escape_env",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        # List your project's dependencies here
        # e.g., "requests>=2.20.0",
    ],
    author="Frank Shih",
    author_email="fshih37@gmail.com",
    description="Escape Env experiments",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/FrankShih0807/escape_env",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)