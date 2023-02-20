from setuptools import setup

setup(
    name='spark-sphinx-theme',
    version='0.0.1',
    packages=['spark_sphinx_theme'],
    include_package_data=True,
    entry_points={"sphinx.html_themes": ["spark_sphinx_theme = spark_sphinx_theme"]},
    install_requires=['pydata_sphinx_theme'],
    description='A Spark sphinx theme',
)
