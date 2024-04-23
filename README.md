# RAG-Pipeline-Q-A-Docs

**Create and run a local RAG pipeline from scratch**

The goal of this notebook is to build a RAG (Retrieval Augmented Generation) pipeline from scratch and have it run on a local GPU.
Specifically, we'd like to be able to open a PDF file, ask questions (queries) of it and have them answered by a Large Language Model (LLM).
There are frameworks that replicate this kind of workflow, including LlamaIndex and LangChain, however, the goal of building from scratch is to be able to inspect and customize all the parts.

**What is RAG?**
RAG stands for Retrieval Augmented Generation.
Each step can be roughly broken down to:

**Retrieval** - Seeking relevant information from a source given a query. For example, getting relevant passages of Wikipedia text from a database given a question.

**Augmented** - Using the relevant retrieved information to modify an input to a generative model (e.g. an LLM).

**Generation** - Generating an output given an input. For example, in the case of an LLM, generating a passage of text given an input prompt.




**Key terms**

**Token** A sub-word piece of text. A token can be a whole word, part of a word or group of punctuation characters. 1 token ~= 4 characters in English, 100 tokens ~= 75 words. Text gets broken into tokens before being passed to an LLM.

**Embedding** A learned numerical representation of a piece of data. For example, a sentence of text could be represented by a vector with 768 values. Similar pieces of text (in meaning) will ideally have similar values.

**Embedding model** A model designed to accept input data and output a numerical representation. For example, a text embedding model may take in 384 tokens of text and turn it into a vector of size 768. An embedding model can and often is different to an LLM model.

**Similarity search/vector search** Similarity search/vector search aims to find two vectors which are close together in high-demensional space. For example, two pieces of similar text passed through an embedding model should have a high similarity score, whereas two pieces of text about different topics will have a lower similarity score. Common similarity score measures are dot product and cosine similarity.

**Large Language Model (LLM)** A model which has been trained to numerically represent the patterns in text. A generative LLM will continue a sequence when given a sequence. For example, given a sequence of the text "hello, world!", a genertive LLM may produce "we're going to build a RAG pipeline today!". This generation will be highly dependant on the training data and prompt.

**LLM context window** The number of tokens a LLM can accept as input. A higher context window means an LLM can accept more relevant information to assist with a query. For example, in a RAG pipeline, if a model has a larger context window, it can accept more reference items from the retrieval system to aid with its generation.

**Prompt** A common term for describing the input to a generative LLM. The idea of "prompt engineering" is to structure a text-based (or potentially image-based as well) input to a generative LLM in a specific way so that the generated output is ideal. This technique is possible because of a LLMs capacity for in-context learning, as in, it is able to use its representation of language to breakdown the prompt and recognize what a suitable output may be .


**How retrieval augmented generation works**

Following are the high level steps needed for the implementation for retrieval augmented generation.

1. Open and Extract text from PDF.
2. Index the extracted text, often as vector embeddings and store.
3. Let the user ask questions related to the source. Perform a similarity search in the index and retrieve relevant text chunks.
4. Insert these text chunks in the prompt along with the question.
5. Request an LLM (e.g. chatgpt) to produce an answer only based on the context

**Instructions**

1. Clone the repo locally
   
   	git clone https://github.com/Rubal-Mahajan/RAG-Pipeline-Q-A-Docs.git

2. Install all the requirements in any of the python venv
   
      pip install -r requirements.txt

3. Add the relevant pdf in data folder and generate the embeddings first with following command
   
      python run embeddings.py

