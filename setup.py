from setuptools import setup, find_packages


setup(
    name='xpdtools',
    version='0.0.0',
    packages=find_packages(),
    description='data processing module',
    zip_safe=False,
    # package_data={'xpdan': ['config/*']},
    include_package_data=True,
    entry_points={'console_scripts': 'image_to_iq = xpdtools.raw_to_iq:main2'}
)
