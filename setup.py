from setuptools import setup, find_packages
from typing import List

def get_requirenments(file_path: str) -> List[str]:
    '''
    This function will return a list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n', '') for req in requirements]
        if '-e .' in requirements:
            requirements.remove('-e .')
    return requirements

setup(
    name='CTR-Prediction',
    version='0.0.1',
    author='Vikram Karthikeyan',
    author_email='viki.kma085@gmail.com',
    packages=find_packages(),
    install_requires=get_requirenments('requirements.txt')
)