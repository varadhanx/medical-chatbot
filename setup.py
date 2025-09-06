from setuptools import find_packages, setup

setup(
    name="medical_chatbot",
    version="0.1.0",
    author="Boktiar Ahmed Bappy",
    author_email="entbappy73@gmail.com",
    description="An AI-powered medical chatbot that helps patients and hospitals find resources and information.",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "sentence-transformers",
        "torch",
        "transformers",
        "scikit-learn",
        "numpy",
        "pandas",
        "requests"
    ],
    python_requires=">=3.7",
)
