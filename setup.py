from setuptools import setup
from setuptools import find_packages


setup(
    name='aorun',
    version='0.1',
    description='Deep Learning over PyTorch',
    author='Ramon Oliveira',
    author_email='ramon@roliveira.net',
    url='https://github.com/ramon-oliveira/aorun',
    license='MIT',
    keywords='neural-networks deep-learning pytorch',
    install_requires=[
        'numpy>=1.10',
        'tqdm>=4.11',
        'requests>=2.12',
        'torch>=0.1.10',
    ],
    extra_requires={
        'tests': [
            'pytest',
            'pytest-cov',
            'pytest-pep8',
        ]
    },
    packages=find_packages(),
    classifiers=[
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.6',
        'Development Status :: 3 - Alpha',
    ]
)
