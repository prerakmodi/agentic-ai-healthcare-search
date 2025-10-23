
from sentence_transformers import SentenceTransformer
import embeddings


# Uses mpnetv2 right now for querying but can just change it.
model3 = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
embeddings3 = model3.encode(embeddings.sentences)

#To exit the query, type exit
#Wait for it to query, its encoding first so...

if __name__ == "__main__":
	print("Type your query and get the top 3 most similar content items. Type 'exit' to quit.")
	while True:
		user_query = input("\nEnter your query: ")
		if user_query.strip().lower() == 'exit':
			break
		top3 = embeddings.get_top_k_similar_content(user_query, embeddings3, embeddings.sentences, model3, k=3)
		print("Top 3 most similar content:")
		for i, content in enumerate(top3, 1):
			print(f"{i}. {content}")