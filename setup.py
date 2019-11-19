from distutils.core import setup

setup(name='PcafeaturesD3MWrapper',
    version='3.1.2',
    description='A wrapper for running the punk pcafeatures functions in the d3m environment.',
    packages=['PcafeaturesD3MWrapper'],
    install_requires=["numpy",
        "pandas==0.25.2",
        "punk==3.0.0"],
    dependency_links=[
    ],
    entry_points = {
        'd3m.primitives': [
            'feature_selection.pca_features.Pcafeatures = PcafeaturesD3MWrapper:pcafeatures'
        ],
    },
)
