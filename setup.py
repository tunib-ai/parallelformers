# Copyright 2021 TUNiB inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup, find_packages

install_requires = [
    'transformers>=4.2',
    'torch',
    "dacite",
    "dataclasses;python_version<'3.7'"
]

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='parallelformers',
    version='1.2.4',
    description=
    'An Efficient Model Parallelization Toolkit for Deployment',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='TUNiB',
    author_email='contact@tunib.ai',
    url='https://github.com/tunib-ai/parallelformers',
    install_requires=install_requires,
    packages=find_packages(),
    package_data={},
    zip_safe=False,
    python_requires=">=3.6.0",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries',
    ],
)
