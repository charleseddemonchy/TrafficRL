from setuptools import setup, find_packages

setup(
    name="traffic_rl",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "matplotlib",
        "gymnasium",
        "pygame",
        "pandas",
        "seaborn",
    ],
    entry_points={
        "console_scripts": [
            "traffic_rl=traffic_rl.cli:main",
        ],
    },
    python_requires=">=3.7",
    description="Reinforcement Learning for Traffic Light Control",
    author="Traffic RL Team",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)