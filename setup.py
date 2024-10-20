import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = 'PyRAG',
    version= '1.1',
    author= 'Sulaiman Shamasna',
    author_email= 'suleiman.shamasneh@gmail.com',
    url= 'https://github.com/sulaiman-shamasna/PyRAG',
    description= 'xx',
    packages=['pyrag', 'pyrag/examples'],
    long_description=read('README.md'),
    entry_points = {
        'console_scripts': [
            'query_transformation=pyrag.examples.query_transformation:main',
            'rag_system=pyrag.examples.rag_system:main',
        ]
    },
    install_requires=[
        'numpy'

    ]

)