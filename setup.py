from setuptools import setup, find_packages


setup(
    name="xpdtools",
    version='0.4.3',
    packages=find_packages(),
    description="data processing module",
    zip_safe=False,
    # package_data={'xpdan': ['config/*']},
    include_package_data=True,
    entry_points={
        "console_scripts": "image_to_iq = xpdtools.cli.process_tiff:run_main"
    },
)
