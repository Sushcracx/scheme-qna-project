Here’s the basic flow of how this project gets scheme_qna:

1.Get the Data: First, we need info about the schemes. You can run the scraper script scraper.py, or just use the `Data.json` file already included in data folder.
2.Understand the Schemes When you start the app (`model+app.py`), it reads the scheme data. It then uses an AI model (`all-MiniLM-L6-v2`) to figure out the meaning of the text for each scheme (like its description, who it's for, benefits, etc.). It turns this understanding into a list of numbers called an "embedding".
3.Create a Search Index:All these embeddings are organized into a special search index using FAISS. This lets the app quickly find schemes with similar meanings later on.
4.Ask a Question: You type your question into the Gradio app.
5.Understand the Question: The same AI model reads your question and creates an embedding for it, too.
6.Find Matches: The app uses your question's embedding to search the FAISS index for the scheme embeddings that are the closest match in meaning.
7.Show Results: The app looks up the details for the best-matching schemes and displays them for you.