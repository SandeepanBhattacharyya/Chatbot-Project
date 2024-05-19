#1. INSTALLING RELEVANT PACKAGES
Streamlit / PyPDF2/ langChain / Faiss-cpu / openAI

#2. CREATE TWO FILES APP.PY AND LOGS.TXT
#3. OVERALL APPROACH

OPENAI_API_KEY=" your API KEY"

-- upload pdf files
-- Extracting the text from pdf files using PdfReader.
-- breaking the text into chunks. Text splitter will be used from langchain.
-- post generaton of the chunks, we will create those embeddings and then store them in a vector store this is like a database to store the embeddings.
-- creating our vector store using FAISS - Facebook AI Semantic Search, Again Langchain will be used.
-- Fine Tuning
-- We will proceed to Finetuning sectiom. - 1. get user's question/ 2. do similarity search/ 3. output results
-- definine our LLM: - gpt 3-5 turbo
-- Procced towards uploading files.

#4. WHILE RUNNING FROM COLAB WE CAN USE THE BELOW CODE SNIPPETS TO CHECK:
!npm install localtunnel/ !streamlit run app.py &> logs.txt & curl ipv4.icanhazip.com/ !npx localtunnel --port (port)

#5. Once the App is live in Streamlit application, we can upload PDF files to get the answers to our custom questions.
