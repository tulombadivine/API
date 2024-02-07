from fastapi import FastAPI,HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from lime.lime_tabular import LimeTabularExplainer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import mlflow.pyfunc
import joblib
import pandas as pd
import numpy as np
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from fastapi.responses import JSONResponse
import json
import logging

# Chargement model
model_uri = "./mlruns/doc/artifacts/model"
model = mlflow.sklearn.load_model(model_uri)
# Local


# Extraire le préprocesseur
preprocessor = model.named_steps['preprocessor']
app = FastAPI()

# data
full_df_predict = pd.read_csv("./data/application_test_cleaned.csv",index_col = 0)
adress_client = pd.read_csv("./data/adress_client.csv",index_col = 0)
# data preprocess
full_df_predict_transformed = preprocessor.transform(full_df_predict)

# Obtenir les transformateurs du préprocesseur
transformers = preprocessor.named_transformers_

# Preprocessing des colonnes
if isinstance(preprocessor, ColumnTransformer):
    num_cols = []
    cat_cols = []

    # Extraction des noms de colonnes pour chaque transformateur
    for name, transformer, column in preprocessor.transformers_:
        if name == 'num':
            num_cols.extend(column)  # Ajouter les noms des colonnes numériques
        elif name == 'cat':
            # Pour OneHotEncoder, obtenir les noms de colonnes modifiés
            if column:  # Si la liste des colonnes catégorielles n'est pas vide
                if hasattr(transformer, 'named_steps') and 'onehot' in transformer.named_steps:
                    onehot_encoder = transformer.named_steps['onehot']
                    cat_cols.extend(onehot_encoder.get_feature_names_out(column))
                else:
                    cat_cols.extend(column)  # Ajoute les noms des colonnes catégorielles
    all_columns = np.concatenate([num_cols, cat_cols])


# Crée un DataFrame avec les données transformées
full_df_predict_transformed_df = pd.DataFrame(full_df_predict_transformed, columns=all_columns)

# Initialise l'Explainer LIME (peut être fait dans un bloc de démarrage ou global)
explainer = LimeTabularExplainer(full_df_predict_transformed_df.values, 
                                 feature_names=full_df_predict_transformed_df.columns, 
                                 class_names=['0', '1'], 
                                 verbose=True, 
                                 mode='classification',
                                 discretize_continuous=False)

full_df_predict_transformed_df['SK_ID_CURR']=full_df_predict['SK_ID_CURR'].tolist()

# Fonction de prédiction personnalisée pour LIME
def custom_predict_fn(data_as_np_array):
    data_as_df = pd.DataFrame(data_as_np_array, columns=[col for col in full_df_predict_transformed_df.columns if col != 'SK_ID_CURR'])
    return model.predict_proba(data_as_df)

# Class pour d'identification du client
class ClientRequest(BaseModel):
    client_id: int

# Encoder pour bug encodage
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return None
        return super().default(obj)

# Fonction permettant de lister tout les ID du client
@app.get('/list_client')
def list_client():
    all_client = full_df_predict.SK_ID_CURR.to_list()
    return all_client

# Fonction de recherche des infos clients
@app.post('/client_adress')
async def client_adress(request: ClientRequest):
     client_data_ad = adress_client[adress_client['SK_ID_CURR'] == request.client_id]
     fname = client_data_ad["First Name"].values[0]
     lname = client_data_ad["Last Name"].values[0]
     adclient = client_data_ad["Address"].values[0]

     return {
         "client_id" : request.client_id,
         "first_name":fname,
         "last_name": lname,
         "adress": adclient
     }

# Fonction de prédiction de crédit
@app.post('/predict_for_client')
async def predict_for_client(request: ClientRequest):
    # Charger les données du client spécifique ici
    client_data = full_df_predict[full_df_predict['SK_ID_CURR'] == request.client_id]
    client_data_LIME = full_df_predict_transformed_df[full_df_predict_transformed_df['SK_ID_CURR'] == request.client_id]

    # Log pour débogage
    print(full_df_predict_transformed_df)
    print(f"Nombre d'enregistrements trouvés dans client_data: {len(client_data)}")
    print(f"Nombre d'enregistrements trouvés dans client_data_LIME: {len(client_data_LIME)}")

    if client_data.empty or client_data_LIME.empty:
        return {"error": "Client ID not found or no data available"}
    
    client_data_LIME = client_data_LIME.drop(['SK_ID_CURR'], axis=1).copy()

    # Effectue la prédiction
    prediction = model.predict(client_data)

    # Calcule l'explication LIME pour le client spécifique
    exp = explainer.explain_instance(client_data_LIME.values[0], custom_predict_fn)
    right_score = exp.predict_proba[1]
    right_score = float(right_score)

    # Extrait les importances locales et les noms de caractéristiques
    local_importance = exp.as_list()

    # Crée un DataFrame pour l'importance locale
    local_importance_df = pd.DataFrame(local_importance, columns=["Feature", "Importance"])

    # Vérifie et remplacer les valeurs flottantes non conformes dans right_score et local_importance_df
    right_score = None if np.isnan(right_score) or np.isinf(right_score) else float(right_score)
    local_importance_df = local_importance_df.applymap(lambda x: None if isinstance(x, float) and (np.isnan(x) or np.isinf(x)) else x)

    # Génère le graphique d'importance locale
    fig, ax = plt.subplots()
    bars = ax.barh(local_importance_df['Feature'], local_importance_df['Importance'], color=np.where(local_importance_df['Importance'] < 0, 'green', 'red'))

    # Paramètres de style
    ax.set_facecolor('none')  # Fond transparent
    plt.setp(ax.get_xticklabels(), fontweight='bold', color='white')  # Écriture grasse et blanche pour les ticks x
    plt.setp(ax.get_yticklabels(), fontweight='bold', color='white')  # Écriture grasse et blanche pour les ticks y
    ax.xaxis.label.set_color('white')  # Couleur de l'étiquette axe x
    ax.yaxis.label.set_color('white')  # Couleur de l'étiquette axe y
    ax.title.set_color('white')  # Couleur du titre
    plt.gca().invert_yaxis()  # Inverser l'axe y pour avoir l'importance du haut vers le bas

    # Maj des légendes 
    if ax.get_legend() is not None:
        plt.setp(ax.get_legend().get_texts(), color='white')

    # Enregistre le graphique avec un fond transparent
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', transparent=True)
    buf.seek(0)
    image_png = buf.getvalue()
    buf.close()

    # Ferme la figure pour libérer la mémoire
    plt.close(fig)

    # Encode l'image en base64 pour la transmission via API
    encoded_image = base64.b64encode(image_png).decode("utf-8")
    try:
        response_data = {
            "lime_importance_plot": encoded_image,
            "right_score": right_score,
            "client_id": request.client_id,
            "prediction": prediction.tolist(),
            "local_importance": local_importance_df.to_dict()
        }

        response_json = json.dumps(response_data, cls=CustomJSONEncoder)
        return JSONResponse(content=json.loads(response_json))
    
    except Exception as e:
        return {"error": str(e)}

# Fonction pour récupérer une image
@app.get("/image/{image_name}")
async def get_image(image_name: str):
    file_path = f"./{image_name}.png"
    try:
        return FileResponse(file_path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Image not found")

# Fonction pour avoir le résultat de prédiction de tout les clients (sans détail)
@app.get('/prediction_for_all')
async def prediction_for_all():
    # copy du dataset
    data = full_df_predict.copy()
    list_var = full_df_predict_transformed_df.columns.tolist()
    elements_to_remove = ["Unnamed: 0", "index", "SK_ID_CURR", "TARGET"]
    for element in elements_to_remove:
        if element in list_var:
            list_var.remove(element)
    # Predict 
    prediction = model.predict(data)
    # Copy df
    df_final_final_viz = full_df_predict_transformed_df.copy()
    # Ajout des predictions
    df_final_final_viz["Prediction"] = prediction
    df_final_final_viz["Prediction"] = df_final_final_viz["Prediction"].astype('str')
    df_final_final_viz["SK_ID_CURR"] = df_final_final_viz["SK_ID_CURR"].astype('str')

    return{
         "data_viz" : df_final_final_viz,
         "list_var":list_var,
     }