import fitz
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import pytesseract
from pytesseract import Output
import pandas as pd
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader
import chromadb
import json

folder = 'BankStatement'
file_name = 'BS1';
pdf_path = f'/content/{folder}/{file_name}.pdf';
images_folder_path = f'/content/{folder}/{file_name}Images';
texts_folder_path = f'/content/{folder}/{file_name}Texts';
os.makedirs(new_images_folder_path);
os.makedirs(texts_folder_path);
doc = fitz.open(pdf_path)
total_pages = doc.page_count + 1

#Extract images from pdf (apply filters)
for page in range(1, total_pages):
  pix = page.get_pixmap(matrix=fitz.Identity, dpi=300, colorspace=fitz.csGRAY, clip=None, annots=True)

  # Convert fitz Pixmap to NumPy array (OpenCV format)
  image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)

    # Apply Adaptive Thresholding for binary image
  _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply Gaussian Blur for denoising
  denoised_image = cv2.GaussianBlur(binary_image, (5, 5), 0)

    # Convert NumPy array back to PIL Image
  enhanced_image = Image.fromarray(denoised_image)

    # Enhance contrast
  enhancer = ImageEnhance.Contrast(enhanced_image)
  contrast_enhanced_image = enhancer.enhance(1.2)  # Experiment with enhancement factor

    # Save the pre-processed image as PNG
  image_path = os.path.join(images_folder_path, f"image{page}-{file_name}.jpg")
  contrast_enhanced_image.save(image_path)

    # Close the pixmap object
  pix = None


#Extract text from the images using pytesseract library
for page_number in range(1, total_pages):
  image_path = os.path.join(images_folder_path, f"image{page}-{file_name}.jpg")
  image = Image.open(image_path)
  img = cv2.imread(image_path)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

  custom_config = r'--oem 3 --psm 6 outputbase whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-~'
  d = pytesseract.image_to_data(thresh, config=custom_config, output_type=Output.DICT)
  df = pd.DataFrame(d)

  df1 = df[(df.conf != '-1') & (df.text != ' ') & (df.text != '')]
  pd.set_option('display.max_rows', None)
  pd.set_option('display.max_columns', None)

  sorted_blocks = df1.groupby('block_num').first().sort_values('top').index.tolist()
  for block in sorted_blocks:
      curr = df1[df1['block_num'] == block]
      sel = curr[curr.text.str.len() > 3]
      # sel = curr
      char_w = (sel.width / sel.text.str.len()).mean()
      prev_par, prev_line, prev_left = 0, 0, 0
      text = ''
      for ix, ln in curr.iterrows():
          # add new line when necessary
          if prev_par != ln['par_num']:
              text += '\n'
              prev_par = ln['par_num']
              prev_line = ln['line_num']
              prev_left = 0
          elif prev_line != ln['line_num']:
              text += '\n'
              prev_line = ln['line_num']
              prev_left = 0

          added = 0  # num of spaces that should be added
          if ln['left'] / char_w > prev_left + 1:
              added = int((ln['left']) / char_w) - prev_left
              text += ' ' * added
          text += ln['text'] + ' '
          prev_left += len(ln['text']) + added + 1
      text += '\n'
      print(text)

  #write the text in txt file
  try:
      # Open the file in write mode and create it if it doesn't exist
      text_path = os.path.join(texts_folder_path, f"Text{page_number}-{file_name}.txt")
      with open(text_path, 'w') as file:
          file.write(text)
      print(f"Successfully created and wrote the text.")
  except Exception as e:
      print(f"An error occurred: {e}")

  
result_arr = []
#Loop over all the pages and extract transactions from each txt file
for page_number in range(1, total_pages):
  print("Page Number: ", page_number)
  client = chromadb.Client()
  collection = client.get_collection(name="langchain")
  collection.delete();
  text_path = os.path.join(texts_folder_path, f"Text{page_number}-{file_name}.txt")
  raw_documents = TextLoader(text_path).load()
  # split the documents into chunks
  text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
  texts = text_splitter.split_documents(raw_documents)

  embeddings = OpenAIEmbeddings()
  db2 = Chroma.from_documents(texts, embeddings)
  # expose this index in a retriever interface
  retriever2 = db2.as_retriever(search_type="similarity")

  template = """Use the provided bank statement text to extract the transactions from it. Your task is to extract bank statement from the text in the specified format.

  Bank Statement Text : {context}

  Extract the following information from the bank statement text:

  1. Date: The date of the transaction in the format M/DD (e.g., 3/27 or 4/10).
  2. Title: The title of the transaction, including the card number and the date (if mentioned) in the format M/DD/YY, the other text, do not include the amount.
  3. Amount: The debit or credit amount, including any characters mentioned (e.g., CR, Y, -) Do not put the - sign before the amount.
  4. Balance: The remaining balance, including any characters mentioned.
  5. isValid : if any field is empty, isValid should be set to "false" else "true".

  However if the text provided does not include any specific transactions, it only provides daily balance information or summary by transaction, keep all the fields empty and set isValid to 'false'.

  `Strictly do not add/return any extra text or explanation in the response, only send the extracted bank statement as response in the format specified.`

  Format the output as an array of JSON objects, with the following keys:

  date
  title
  amount
  balance
  isValid

  If a single transaction is present, still format it as an array.

  For example, Input:- 2/27  POS DEB  02/24/17 44900004                     15.03-               691.08
                              7-ELEVEN
                              Card# 5454
  Output:- `{{
              "date" :  "2/27",
              "title" : "POS DEB  02/24/17 44900004 7-ELEVEN Card# 5454",
              "amount" : "15.03-",
              "balance" : "691.08",
              "isValid" : "true"
            }}`

  question : {question}
  """
  from langchain.prompts import ChatPromptTemplate
  PROMPT = PromptTemplate(
      template=template, input_variables=["context", "question"]
  )
  chain_type_kwargs = {"prompt": PROMPT}
  qa4 = RetrievalQA.from_chain_type(llm=ChatOpenAI(model = "gpt-4"), chain_type="stuff", retriever=retriever2, chain_type_kwargs=chain_type_kwargs)

  query = "Please provide me a list of all the transactions."
  result =  qa4.run(query)
  print(result)
  # print(qa4.query)
  # print(result["result"])
  # print(type(result["result"]))
  # src_docs = result["source_documents"]
  # for doc in src_docs:
  #   print(doc, "/n")

  start_index = result.find('[')
  end_index = result.rfind(']')
  final_result = '';
  if start_index != -1 and end_index != -1:
    final_result = result[start_index : end_index + 1].strip()
  else:
    final_result = [{
      "date" :  "",
      "title" : "",
      "amount" : "",
      "balance" : "",
      "isValid" : "false"
    }]
    final_result = json.dumps(final_result)

  data_array = json.loads(final_result)
  for obj in data_array:
    result_arr.append(obj)
    
  client = chromadb.Client()
  collection = client.get_collection(name="langchain")
  collection.delete();

  print(result_arr)

  # with get_openai_callback() as cb:
  #   qa.run(query)
  #   print(cb)


json_path = f'/content/{folder}/{file_name}.json'
with open(json_path, 'w', encoding='utf-8') as inputfile:
  json.dump(result_arr, inputfile)

#Convert json to csv
csv_path = f'/content/BankStatement/{folder}/{file_name}.csv'
json_path = f'/content/BankStatement/{folder}/{file_name}.json'
with open(json_path, encoding='utf-8') as inputfile:
    data = json.load(inputfile)

# Convert each object to the new format
new_data = []
for item in data:
  try:
    if item['isValid'] == 'true':
      jsonObj = {
          'Date' : item['date'],
          'Title' : item['title'],
          'Amount': item['amount'],
          'Balance': item['balance']
      }
      new_data.append(jsonObj)
  except:
    continue

# Write the new JSON data to a file
with open(json_path, 'w', encoding='utf-8') as outputfile:
    json.dump(new_data, outputfile)

with open(json_path, encoding='utf-8') as inputfile:
     df = pd.read_json(inputfile)

df.to_csv(csv_path, encoding='utf-8', index=False)
