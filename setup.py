from setuptools import setup

setup(name='deepsith',
      version='0.1',
      description='DeepSITH: Scale-Invariant Temporal History Across Hidden Layers',
      url='https://github.com/beegica/SITH-Con',
      license='Free for non-commercial use',
      author='Computational Memory Lab',
      author_email='bgj5hk@virginia.edu',
      packages=['deepsith'],
      install_requires=[
          'torch>=1.1.0',
          
      ],
      zip_safe=False)