from setuptools import setup

setup(name='mutual_info',
      version='0.0',
      description="Putting GaelVaraquaux's gist in a repo. Mutual information calculation utils.",
      install_requires=[
          'numpy',
          'scipy',
          'sklearn',
      ],
      packages=['mutual_info'],
      )
