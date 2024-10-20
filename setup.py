import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = 'PyRAG',
    version= '1.0',
    author= 'Sulaiman Shamasna',
    author_email= 'suleiman.shamasneh@gmail.com',
    url= 'https://github.com/sulaiman-shamasna/PyRAG',
    description= 'xx',
    packages=['xx'],
    long_description=read('README.md'),
    entry_points = {
        'console_scripts': [
            'method1=pyrag.method1:main',
            'method2=pyrag.method2:main'
        ]
    },
    install_requires=[
        'torch',
        'X',
        '...'

    ]

)