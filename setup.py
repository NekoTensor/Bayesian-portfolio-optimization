from setuptools import setup, find_packages

setup(
    name='portfolio-optimization',
    version='0.1.0',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'skopt',
        'yfinance',
        'plotly',
        'dash',
        'seaborn',
        'matplotlib',
        'pymc3'
    ],
    author='NekoTensor',
    author_email='nekotensor@gmail.com',
    description='Portfolio Optimization with Bayesian Methods.',
)
