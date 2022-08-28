from setuptools import setup, find_packages

VERSION = '0.1.1'
DESCRIPTION = 'Personalized Text Classification dataset'
LONG_DESCRIPTION = 'Personalized Text Classification dataset with transient labels inspired by Myca productivity tool'

setup(
    name='myca',
    version=VERSION,
    license='MIT',
    author='Yuzhao Stefan Heng',
    author_email='stefan.hg@outlook.com',
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    url='https://github.com/StefanHeng/Personalized-Productivity-Dataset',
    download_url='https://github.com/StefanHeng/Personalized-Productivity-Dataset/archive/refs/tags/v0.1.1.tar.gz',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'pandas',
        'tqdm',
        'stefutils'
    ],
    keywords=['python', 'nlp', 'deep-learning', 'text-classification', 'dataset'],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: MacOS X',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Science/Research',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 3',
        'Topic :: Database'
    ]
)
