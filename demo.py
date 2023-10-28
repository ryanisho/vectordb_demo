"""
This demo is taken from Chroma - ('https://trychroma.com/')

=== What are embeddings? ===

Embeddings are the A.I-native way to represent any kind of data, making them 
the perfect fit for working with all kinds of A.I-powered tools and algorithms. 
They can represent text, images, and soon audio and video.

To create an embedding, data is fed into an embedding model, which outputs 
vectors of numbers. The model is trained in such a way that 'similar' data, e.g. 
text with similar meanings, or images with similar content, will produce vectors 
which are nearer to one another, than those which are dissimilar.

=== Embeddings and retrieval ===

We can use the similarity property of embeddings to search for and retrieve 
information. For example, we can find documents relevant to a particular topic, 
or images similar to a given image. Rather than searching for keywords or tags, 
we can search by finding data with similar semantic meaning.

=== Example Dataset === 

As a demonstration we use the [SciQ dataset](https://arxiv.org/abs/1707.06209),
available from [HuggingFace](https://huggingface.co/datasets/sciq).

=== Dataset description, from HuggingFace: === 

> The SciQ dataset contains 13,679 crowdsourced science exam questions about 
Physics, Chemistry and Biology, among others. The questions are in multiple-choice 
format with 4 answer options each. For the majority of the questions, an additional 
paragraph with supporting evidence for the correct answer is provided.
"""

# Get the SciQ dataset from HuggingFace
import chromadb
from datasets import load_dataset

dataset = load_dataset("sciq", split="train")

print("Number of questions in dataset: ", len(dataset))

# Filter the dataset to only include questions with a support
# "support" - is essentially our answer
dataset = dataset.filter(lambda x: x["support"] != "")

print("Number of questions with support: ", len(dataset))

"""
Loading the data into Chroma

Chroma comes with a built-in embedding model, which makes it simple to load text.
We can load the SciQ dataset into Chroma:
"""

# Import Chroma and instantiate a client.
client = chromadb.Client()

# Create a new Chroma collection to store the supporting evidence.
# We don't need to specify an embedding fuction, and the default will be used.
collection = client.create_collection("sciq_supports")

# Embed and store the first 100 supports for this demo
collection.add(
    ids=[str(i) for i in range(0, 100)],  # IDs are just strings
    documents=dataset["support"][:100],
    metadatas=[{"type": "support"} for _ in range(0, 100)],
)

"""
# Querying the data

Once the data is loaded, we can use Chroma to find supporting evidence for the questions in the dataset.
In this example, we retrieve the most relevant result according to the embedding similarity score.

Chroma handles computing similarity and finding the most relevant results for you, 
so you can focus on building your application.
"""

results = collection.query(
    query_texts=dataset["question"][:10],
    n_results=1
)

# Print the question and the corresponding support
for i, q in enumerate(dataset['question'][:10]):
    print(f"Question: {q}")
    print(f"Retrieved support: {results['documents'][i][0]}")
    print()
