from sentence_transformers import SentenceTransformer
import torch
import gradio as gr
from huggingface_hub import InferenceClient
client = InferenceClient("Qwen/Qwen2.5-7B-Instruct")

with open("knowledge.txt", "r", encoding="utf-8") as file:
  knowledge = file.read()

def preprocess_text(text):

  cleaned_text = text.strip()

  chunks = cleaned_text.split("\n")

  cleaned_chunks = []

  # Write your for-in loop below to clean each chunk and add it to the cleaned_chunks list
  for chunk in chunks:
    stripped_chunks = chunk.strip()
    if len(stripped_chunks) > 0:
      cleaned_chunks.append(stripped_chunks)

  # Return the cleaned_chunks
  return cleaned_chunks

# Call the preprocess_text function and store the result in a cleaned_chunks variable
cleaned_chunks = preprocess_text(knowledge)

model = SentenceTransformer('all-MiniLM-L6-v2')

def create_embeddings(text_chunks):
  # Convert each text chunk into a vector embedding and store as a tensor
  chunk_embeddings = model.encode(text_chunks, convert_to_tensor=True) # Replace ... with the cleaned_chunks list

  # Return the chunk_embeddings
  return chunk_embeddings

# Call the create_embeddings function and store the result in a new chunk_embeddings variable
chunk_embeddings = create_embeddings(cleaned_chunks)

def get_top_chunks(query, chunk_embeddings, text_chunks):
  # Convert the query text into a vector embedding
  query_embedding = model.encode(query, convert_to_tensor=True) # Complete this line

  # Normalize the query embedding to unit length for accurate similarity comparison
  query_embedding_normalized = query_embedding / query_embedding.norm()

  # Normalize all chunk embeddings to unit length for consistent comparison
  chunk_embeddings_normalized = chunk_embeddings / chunk_embeddings.norm(dim=1, keepdim=True)

  # Calculate cosine similarity between all chunks and the query using matrix multiplication
  similarities = torch.matmul(chunk_embeddings_normalized, query_embedding_normalized) # Complete this line


  # Find the indices of the 3 chunks with highest similarity scores
  top_indices = torch.topk(similarities, k=3).indices

  # Create an empty list to store the most relevant chunks
  top_chunks = []

  # Loop through the top indices and retrieve the corresponding text chunks
  # This is only one way scholars may write this, but there are other ways!
  for i in top_indices:
    chunk = text_chunks[i]
    top_chunks.append(chunk)

  # ===== SPICY CHALLENGE: LIST COMPREHENSION =====
  # top_chunks = [chunks[i] for i in top_indices]

  # Return the list of most relevant chunks
  return top_chunks

#


def respond(message, history):
    messages = [{"role": "system", "content": "You are a friendly chatbot that recommends sustainable restaurants from Austin, Seattle, Chicago, and Los Angeles. Recommend up to 3 restaurants at a time, mention the location of each restaurant, and mention average cost. Keep responses under 200 words. Only include restaurants in the KNOWLEDGE.TXT."}]

    if history:
        messages.extend(history)

    food_top_results = get_top_chunks(message, chunk_embeddings, cleaned_chunks)
    context = "\n".join(food_top_results)

    augmented_message = f"Relevant restaurant info:\n{context}\n\nUser question: {message}"
    messages.append({"role": "user", "content": augmented_message})

    response = client.chat_completion(
        messages,
        max_tokens=300,
        temperature=0.7,
        top_p=0.5
    )
    return response.choices[0].message.content.strip()
    return response.choices[0].message.content.strip()


custom_theme = gr.themes.Soft(
    primary_hue="emerald",
    secondary_hue="fuchsia",
    neutral_hue="lime",
    spacing_size="lg",
    radius_size="lg",
    text_size="lg",
    font=[gr.themes.GoogleFont("IBM Plex Sans"), "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "monospace"]
)


with gr.Blocks() as demo:
    gr.Markdown(
        """
        # 🍀 Nature's Table: Find Hidden Gems!
        Get information about sustainable resuturants near you catered to your dietary needs, cuisine type, and price range!*
        """,
        elem_id="title"
    )

    gr.ChatInterface(
        fn=respond,
        examples=[
            "Give me affordable resutrants in the LA area",
            "I'm visiting Austin. Which resturants should I go to?",
            "I'm a vegan looking for resturants in Seattle.",
            "Can you share some information about hidden gems in Chicago?",
        ]
    )

demo.launch(ssr_mode=False, theme = custom_theme)
