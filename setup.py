from setuptools import setup, find_packages

setup(
    name='speech-articulatory-coding',  
    version='0.1.0',
    author='Cheol Jun Cho',
    author_email='cheoljun@berkeley.edu',
    description='Python code for analyzing and synthesizing articulatory features of speech',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Berkeley-Speech-Group/Speech-Articulatory-Coding',  
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    install_requires=[
        'numpy',
        'soundfile',
        'librosa',
        'torch',
        'torchaudio',
        'transformers',
        'torchcrepe==0.0.23',
        'huggingface-hub',
        'matplotlib',
        'tqdm',
        'penn',
    ],
    include_package_data=False,  
    license='MIT',  
)
