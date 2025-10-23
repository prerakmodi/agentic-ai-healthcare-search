import embeddings
from sentence_transformers import SentenceTransformer, util
import time

#tied between MiniLM and MPNet for fastest encoding, it alternates between the 2 for the fastest everytime i run it.

model = SentenceTransformer('sentence-transformers/LaBSE')
embeddings1 = model.encode(embeddings.sentences)
model2 = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
embeddings2 = model2.encode(embeddings.sentences)
model3 = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
embeddings3 = model3.encode(embeddings.sentences)

if __name__ == "__main__":
    # LaBSE model timing currently at: around 8 seconds encoding
    # 0.0114 seconds similarity run time
    
    #LaBSE Encoding Timing
    start = time.time()
    _ = model.encode(embeddings.sentences)
    encode_time_labse = time.time() - start
    #LaBSE Similarity Timing
    start = time.time()
    similarities = model.similarity(embeddings1, embeddings1)
    similarity_time_labse = time.time() - start

    print(f"LaBSE encoding time: {encode_time_labse:.4f} seconds")
    print(f"LaBSE similarity time: {similarity_time_labse:.4f} seconds")
    print("LaBSE similarities:")
    print(similarities)

    # MiniLM model timing
    # MINILM model timing currently at: around 3 seconds encoding
    # 0.0001 seconds similarity run time
    start = time.time()
    _ = model2.encode(embeddings.sentences)
    encode_time_minilm = time.time() - start

    start = time.time()
    similarities2 = model2.similarity(embeddings2, embeddings2)
    similarity_time_minilm = time.time() - start

    print(f"MiniLM encoding time: {encode_time_minilm:.4f} seconds")
    print(f"MiniLM similarity time: {similarity_time_minilm:.4f} seconds")
    print("MiniLM similarities:")
    print(similarities2)

    # all-mpnet-base-v2 model timing
    # all-mpnet-base-v2 model timing currently at: around 3 seconds encoding
    # 0.0189 seconds similarity run time
    start = time.time()
    _ = model3.encode(embeddings.sentences)
    encode_time_mpnet = time.time() - start

    start = time.time()
    similarities3 = model3.similarity(embeddings3, embeddings3)
    similarity_time_mpnet = time.time() - start

    print(f"MPNet encoding time: {encode_time_mpnet:.4f} seconds")
    print(f"MPNet similarity time: {similarity_time_mpnet:.4f} seconds")
    print("MPNet similarities:")
    print(similarities3)

