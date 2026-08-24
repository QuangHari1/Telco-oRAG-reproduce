# Telco-RAG: Retrieval-Augmented Generation for Telecommunications

## Re-run the local offline baseline

The reproducible, local-3GPP benchmark setup is documented in
[Telco-RAG_api/PAPER_REPRODUCTION.md](Telco-RAG_api/PAPER_REPRODUCTION.md).
It downloads pinned corpus and TeleQnA revisions, runs the two-pass offline
retrieval path, and writes a resume-safe JSONL checkpoint. This is a modified
offline reproduction, not a claim of numerical equivalence to the paper's
tables: its dataset, hosted model, and MCQ scoring prompt are explicitly
documented there.

**Telco-RAG** is a specialized Retrieval-Augmented Generation (RAG) framework designed to address the unique challenges of the telecommunications industry, particularly in handling the complexity and rapid evolution of 3GPP documents.

## References

- Bornea, A.-L., Ayed, F., De Domenico, A., Piovesan, N., & Maatouk, A. (2024). *Telco-RAG: Navigating the Challenges of Retrieval-Augmented Language Models for Telecommunications*. *arXiv preprint arXiv:2404.15939*. [DOI](https://doi.org/10.48550/arXiv.2404.15939) | [Read the paper](https://arxiv.org/pdf/2404.15939.pdf)

## Features

- **Custom RAG Pipeline**: Specifically tailored to handle the complexities of telecommunications standards.
- **Enhanced Query Processing**: Implements a dual-stage query enhancement and retrieval process, improving the accuracy and relevance of generated responses.
- **Hyperparameter Optimization**: Optimized for the best performance by fine-tuning chunk sizes, context length, and embedding models.
- **NN Router**: A neural network-based router that enhances document retrieval efficiency while significantly reducing RAM usage.
- **Open-Source**: Freely available for the community to use, adapt, and improve.

## Presentation Video

![Watch the video](https://github.com/netop-team/Telco-RAG/blob/main/video_720p.gif)

The video is presented at 1.5x speed.

## Getting Started

To get started with **Telco-RAG**, clone the repository and set up the environment:

```bash
git clone https://github.com/netop-team/telco-rag.git
cd telco-rag
```

### Prerequisites

- Python 3.11
- Node.js

Other dependencies are listed in `requirements.txt`.

### Installation

Install the necessary Python packages and download the 3GPP knowledge database:

```bash
cd ./Telco-RAG_api
pip install -r requirements.txt
python setup.py
```

### Running the Full Pipeline

To run the full pipeline, use the following commands:

```bash
npm install
npm run dev
```

This will open two terminals: one for the frontend and one for the Telco-RAG backend. You can access the frontend via your browser at `http://localhost:3000/`.

On your first connection, ensure to specify a valid OpenAI API key in the settings.

### Running Only the API Server

If you only want to run the API server, use this command:

```bash
cd ./Telco-RAG_api
uvicorn api.deploy_api:app --reload
```

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
