# Importamos todas las dependencias requeridas, en este caso será Gradio para desarrollar la interfaz gráfica y openai para realizar los llamados a su API
import openai
import pandas as pd
from flask import Flask, jsonify, request

from openai.embeddings_utils import get_embedding
from openai.embeddings_utils import cosine_similarity
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

app = Flask(__name__)

loader = PyPDFLoader("libro-carlos1cap.pdf")
pages = loader.load_and_split()

# Objeto que va a hacer los cortes en el texto
split = CharacterTextSplitter(chunk_size=300, separator = '.\n')

textos = split.split_documents(pages) # Lista de textos

# Extraemos la parte de page_content de cada texto y lo pasamos a un dataframe
textos = [str(i.page_content) for i in textos] #Lista de parrafos
parrafos = pd.DataFrame(textos, columns=["texto"])

def generar_embeddings():
    parrafos['Embedding'] = parrafos["texto"].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
    parrafos.to_csv('librocarlos.csv')

# La misma funcion del chatbot de pregunts y respuestas
def embed_text(path="texto.csv"):
    conocimiento_df = pd.read_csv(path)
    conocimiento_df['Embedding'] = conocimiento_df['texto'].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
    conocimiento_df.to_csv('mtg-embeddings.csv')
    return conocimiento_df

def buscar(busqueda, datos, n_resultados=5): 
    busqueda_embed = get_embedding(busqueda, engine="text-embedding-ada-002")
    datos["Similitud"] = datos['Embedding'].apply(lambda x: cosine_similarity(x, busqueda_embed))
    datos = datos.sort_values("Similitud", ascending=False)
    return datos.iloc[:n_resultados][["texto", "Similitud", "Embedding"]]

texto_emb = parrafos

@app.route('/embedding', methods=['POST'])
def buscar_handler():
    data = request.json
    busqueda = data['busqueda']
    api_key = data['api_key']
    
    openai.api_key = api_key

    generar_embeddings()
    resultados = buscar(busqueda, texto_emb)

    ##Transformamos el vector de la posicion 0 -> Embeddings a texto
    respuesta = resultados.iloc[0]['texto']

      # Limpiamos el string y eliminamos los saltos de línea
    respuesta = respuesta.replace("\n", "")
    respuesta = respuesta.replace("-", "")

    return jsonify(
    {
        "set_attributes": {
            "resultado": respuesta
        },
        "redirect_to_node": [],
        "messages": [],
        "files": []
    })

@app.route('/', methods=['GET'])
def home():
    return "Servicio en línea para buscar respuestas."


if __name__ == '__main__':
    app.run()