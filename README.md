# Ambiguity Detection in Short Search Queries

This repository contains the implementation and evaluation framework developed as part of a Master's thesis at the University of Bayreuth. The project focuses on detecting ambiguity in short search queries and investigating whether user profile context can improve ambiguity estimation compared to profile-agnostic approaches.

## Research Objectives

The work is guided by the following research questions:

**RQ1:** Given an ambiguous query such as *Amazon* or *Python*, can the ambiguity detection system determine whether the query is ambiguous or unambiguous?

**RQ2:** Does incorporating user profile context reduce the ambiguity score of short queries compared to a profile-agnostic baseline?

**RQ3:** To what extent does lexical ambiguity indicated by external knowledge sources correspond to the ambiguity experienced by individual users?

## Project Overview

Many web search queries are short and lack sufficient context for accurate interpretation. A query such as *Python* may refer to a programming language, a snake species, or other entities. Traditional ambiguity detection methods typically rely on lexical or statistical signals without considering the user's background knowledge or interests.

This project introduces a user-aware ambiguity detection approach that combines:

* Lexical ambiguity analysis
* Semantic similarity measures
* User profile representations
* Context vocabulary extraction
* Knowledge-based ambiguity indicators

The system aims to determine whether a query is ambiguous and how ambiguity changes when user context is taken into account.

## Repository Structure

```text
.
├── src/                     # Source code
├── data/                    # Datasets and input files
├── models/                  # Trained models and embeddings
├── experiments/             # Experiment scripts
├── results/                 # Evaluation outputs
├── tests/                   # Unit tests
├── README.md
└── requirements.txt
```

## Methodology

The ambiguity detection pipeline consists of the following stages:

1. **Query Processing**

   * Tokenization and normalization
   * Query validation

2. **Lexical Ambiguity Analysis**

   * Extraction of ambiguity indicators
   * Sense and entity analysis

3. **User Context Modeling**

   * Construction of user profiles
   * Context vocabulary generation

4. **Semantic Similarity Computation**

   * Embedding-based similarity calculations
   * Context-query relevance estimation

5. **Ambiguity Scoring**

   * Combination of lexical and semantic signals
   * Profile-aware ambiguity adjustment

6. **Evaluation**

   * Ambiguous query dataset
   * Unambiguous query dataset
   * Comparative analysis against baseline methods

## Docker Setup

### Prerequisites
- Install [Docker Desktop](https://www.docker.com/products/docker-desktop/).
- Ensure Docker is running.

### Build and Run

Clone the repository and navigate to the project directory:

```bash
git clone <REPOSITORY_URL>
cd ambiguityDetection
```

Build the Docker images and start the application:

```bash
docker compose up --build
```

To run the application in the background:

```bash
docker compose up --build -d
```

### Access the Application
- **Frontend:** http://localhost:3000
- **Backend API:** http://localhost:8000
- **API documentation:** http://localhost:8000/docs

### Stop the Application

```bash
docker compose down
```

## Installation for local env setup

### Prerequisites

* Python 3.10+
* pip
* Git

### Setup

```bash
git clone <repository-url>
cd ambiguityDetection

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

## Usage

Run the ambiguity detection pipeline:

```bash
python main.py
```

Run experiments:

```bash
python experiments/run_experiments.py
```

Run tests:

```bash
pytest
```

## Datasets

The evaluation uses collections of:

* Ambiguous search queries
* Unambiguous search queries
* User profile data
* External knowledge sources

Dataset files are not included in this repository if they contain proprietary, licensed, or large-scale data.

## Evaluation Metrics

The system may be evaluated using metrics such as:

* Base Score
* Ambiguity Score Distribution
* Profile Impact Analysis

## Thesis Context

This implementation was developed as part of a Master's thesis in Philosophy & Computer Science at the University of Bayreuth.

The research investigates how personalized context influences the interpretation of short search queries and whether ambiguity can be modeled more accurately when user-specific information is available.

## Future Work

Potential extensions include:

* Integration of large language models
* Dynamic user profile adaptation
* Real-time search personalization
* Cross-domain ambiguity detection
* Multilingual query analysis

## Author

**Sumanth Kubasad**

Master's Student
University of Bayreuth
Bayreuth, Germany

## License

This project is provided for academic and research purposes.

