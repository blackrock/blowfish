from setuptools import setup, find_packages

setup(
    name='blowfish',
    version='0.1.0',
    author='Alex De Castro, Thomas-Roland Barillot, Javier Makmuri',
    author_email='Alex.castro@blackrock.com, thomasroland.barillot@blackrock.com, javier.makmuri@blackrock.com',
    description='A Python project focused on vector search, explainability, polysemy, and disambiguation in AI.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/blackrock/blowfish',
    packages=find_packages(),
    install_requires=[
        # List your project's dependencies here.
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: Other/Proprietary License',  # Changed to reflect non-open source status
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'blowfish=blowfish.__main__:main'
        ],
    },
    keywords='vector search, explainability, polysemy, disambiguation',
    project_urls={
        'Bug Reports': 'https://github.com/blackrock/blowfish/issues',
        'Source': 'https://github.com/blackrock/blowfish',
        'How to Contribute': 'https://github.com/blackrock/blowfish/blob/main/CONTRIBUTING.md'
    },
    license='Apache License 2.0',
)
