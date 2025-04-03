from setuptools import setup, find_packages

setup(
    name="mcc_classifier",
    version="0.1.0",
    description="Tool to evaluate MCC classification agents",
    author="InfinitePay Team",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "pandas>=1.0.0",
        "numpy>=1.18.0",
    ],
    extras_require={
        "openai": ["openai>=1.0.0"],
        "dev": ["pytest>=6.0.0", "pytest-cov>=2.10.0"],
    },
    entry_points={
        "console_scripts": [
            "mcc-evaluate=mcc_classifier.main:main",
        ],
    },
) 