import pandas as pd
import streamlit as st
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_validate 
from time import time
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler


#pip install scikit-learn==1.2.2 !!!!!!!! problema do KNN


nome_metricas=['precision', 'recall', 'accuracy']

st.set_page_config(
    page_title= "Machine Learning",
    page_icon= "🧠",
    layout= "wide",
    initial_sidebar_state= "collapsed",
    menu_items= {
        'About': "Desenvolvido por:\n- Dayvson Moura;\n- Thales Mayrinck"
    }
)


def main():
    header()
    df_processado = preprocessamento()
    df_transformado = transformacao(df_processado)
    df_transformado.to_csv("dataframe_final.csv")
    with st.expander("KNeighborsClassifier"):
        kneighbors(df_processado, df_transformado)
    with st.expander("Naive Bayes"):
        naive_bayes(df_processado, df_transformado)
    with st.expander("Árvore de Decisão"):
        arvore(df_processado, df_transformado)
    st.divider()
        


def header():
    st.header("Machine Learning")
    st.markdown("""
                Esta página apresenta os modelos de <i>machine learning</i> criados e os seus resultados.            
                """,
                unsafe_allow_html= True)


@st.cache_data
def ler_dataset():
    df = pd.read_csv("./data/covid_no_mexico.csv")
    return df.copy()


def preprocessamento():
    st.markdown("### Pré-processamento")
    st.markdown("""
                Nesta etapa foram removidas todas as colunas que julgamos não serem necessárias para o Machine Learning.
                """,
                unsafe_allow_html=True)
    df = ler_dataset()
    df = df.drop(["DATA_DE_INGRESSO","DATA_DE_SINTOMAS","DATA_DO_ARQUIVO","DATA_DE_ATUALIZACAO","DATA_DE_OBITO","SETOR","ATRASO","ENTIDADE_UM","ENTIDADE_RES","ENTIDADE_DE_REGISTRO","ENTIDADE","ABR_ENT","ENTIDADE_DE_NASCIMENTO","MUNICIPIO_RES","PAÍS_DE_ORIGEM","PAÍS_DE_NACIONALIDADE","ID_REGISTRO","id","NACIONALIDADE","IDIOMA_INDÍGENA_FALADO"], axis = 1)
    indices_a_remover = np.random.choice(df.index, 253000, replace=False)
    df = df.drop(indices_a_remover)


    if st.checkbox("Mostrar dataset após o pré-processamento dos dados"):
        st.dataframe(df)
    st.divider()
    

    #falta renomear os valores das colunas
    return df





def transformacao(df):
    st.markdown("### Transformação")
    st.markdown("""
                Nesta etapa, as colunas categóricas foram transformadas usando a técnica <i>One-Hot Encoding</i> para converter os seus dados em valores numéricos. Em seguida, foi realizado o balanceamento dos dados utilizando a técnica <i>SMOTE</i> e a normalização da coluna IDADE.
                """,
                unsafe_allow_html=True)
    
    colunas_categoricas_multivalor = ['ORIGEM','SEXO','TIPO_DE_PACIENTE','INTUBADO','PNEUMONIA','GRAVIDEZ','DIABETES','EPOC','ASMA','IMUNOSSUPRIMIDO','HIPERTENSÃO','OUTRAS_COMORBIDADES','DOENÇA_CARDIOVASCULAR','OBESIDADE','DOENÇA_RENAL_CRÔNICA','TABAGISMO','OUTRO_CASO','MIGRANTE','UTI']
    df_transformado = pd.get_dummies(df, columns=colunas_categoricas_multivalor)
    y = df_transformado['RESULTADO'] # Rótulos    
    x = df_transformado.drop('RESULTADO', axis = 1) # Features
    X_res, y_res = SMOTE().fit_resample(x, y)
    df_balanceado = pd.DataFrame(X_res, columns=x.columns)
    df_balanceado['RESULTADO'] = y_res
    scaler = StandardScaler()
    vals = df_balanceado[['IDADE']].values
    df_balanceado['IDADE'] = scaler.fit_transform(vals)
    novo_nome_colunas = {'ORIGEM_1': 'USMER', 'ORIGEM_2': 'FORA DE USMER','ORIGEM_3': 'NÃO ESPECIFICADO','SEXO_1': 'MULHER', 'SEXO_2': 'HOMEM', 'TIPO_DE_PACIENTE_1': 'AMBULATORIO', 'TIPO_DE_PACIENTE_2': 'HOSPITALIZADO', 'TIPO_DE_PACIENTE_3': 'NÃO ESPECIFICADO', 'INTUBADO_1': 'INTUBADO', 'INTUBADO_2': 'NÃO INTUBADO', 'INTUBADO_97': 'NÃO SE APLICA', 'INTUBADO_99': 'NÃO ESPECIFICADO', 'PNEUMONIA_1': 'COM PNEUMONIA', 'PNEUMONIA_2': 'SEM PNEUMONIA', 'PNEUMONIA_99': 'NÃO ESPECIFICADO' , 'GRAVIDEZ_1': 'ESTÁ GRÁVIDA', 'GRAVIDEZ_2': 'NÃO ESTÁ GRÁVIDA','DIABETES_1': 'POSSUI DIABETES', 'DIABETES_2': 'NÃO POSSUI DIABETES', 'DIABETES_98': 'SE IGNORA' , 'EPOC_1': 'POSSUI EPOC', 'EPOC_2': 'NÃO POSSUI EPOC', 'ASMA_1': 'POSSUI ASMA', 'ASMA_2': 'NÃO POSSUI ASMA', 'IMUNOSSUPRIMIDO_1': 'É IMUNOSSUPRIMIDO', 'IMUNOSSUPRIMIDO_2': 'NÃO É IMUNOSSUPRIMIDO','HIPERTENSÃO_1': 'POSSUI HIPERTENSÃO', 'HIPERTENSÃO_2': 'NÃO POSSUI HIPERTENSÃO','OUTRAS_COMORBIDADES_1': 'POSSUI OUTRAS COMORBIDADES', 'OUTRAS_COMORBIDADES_2': 'NÃO POSSUI OUTRAS COMORBIDADES','DOENÇA_CARDIOVASCULAR_1': 'POSSUI DOENÇA CARDIOVASCULAR', 'DOENÇA_CARDIOVASCULAR_2':
    'NÃO POSSUI DOENÇA CARDIOVASCULAR','OBESIDADE_1': 'POSSUI OBESIDADE', 'OBESIDADE_2': 'NÃO POSSUI OBESIDADE','DOENÇA_RENAL_CRÔNICA_1': 'POSSUI DOENÇA RENAL CRÔNICA', 'DOENÇA_RENAL_CRÔNICA_2': 'NÃO POSSUI DOENÇA RENAL CRÔNICA','TABAGISMO_1': 'PRATICA O TABAGISMO','TABAGISMO_2': 'NÃO PRATICA O TABAGISMO', 'OUTRO_CASO_1': 'TEVE CONTATO COM ALGUEM DIAGNOSTICADO COM COVID 19', 'OUTRO_CASO_2': 'NÃO TEVE CONTATO COM ALGUEM DIAGNOSTICADO COM COVID 19', 'MIGRANTE_1': 'É IMIGRANTE', 'MIGRANTE_2': 'NÃO É IMIGRANTE','UTI_1': 'ESTÁ NA UTI', 'UTI_2': 'NÃO ESTÁ NA UTI',}
    df_balanceado.rename(columns=novo_nome_colunas, inplace=True)
    if st.checkbox("Mostrar dataset após a transformação dos dados"):
        st.dataframe(df_balanceado)
    st.divider()
    
    return df_balanceado


    

def validacao_cruzada_por_modelo (modelo, dfinicial, dfprocessado):
    df = {'dfinicial': dfinicial, 'dfprocessado':dfprocessado}
    results_final = []
    for df_nome, dataframe in df.items():
        tempo_randomforest=[]
        divisao, rotulos = dataframe.drop('RESULTADO', axis=1), dataframe['RESULTADO']
        resultados={medida: [] for medida in nome_metricas}
        for i in range (5):
            classifier = modelo
            t1=time()
            execucao = cross_validate(classifier, divisao, rotulos, cv=10, scoring=nome_metricas)
            t2=time()
            tempo_randomforest.append(t2-t1)
            mean_results = {'Dataset': df_nome, 'Mean time': np.mean(tempo_randomforest)}
            for medida in nome_metricas:
                resultados[medida].append(np.mean(execucao[f'test_{medida}']))
                mean_results[medida] = np.mean(resultados[medida])
        results_final.append(mean_results)
    return results_final




def kneighbors(dfinicial, dfprocessado):
    st.markdown("""
                
                O K-Nearest Neighbors (KNN) é um algoritmo de aprendizado de máquina supervisionado usado principalmente para tarefas de classificação e regressão. Ele se baseia no princípio de que exemplos semelhantes estão próximos uns dos outros no espaço de características. O KNN classifica um novo exemplo com base na maioria das classes dos seus vizinhos mais próximos.
                """, 
                unsafe_allow_html=True)
    st.markdown("K=3")
    classifier = KNeighborsClassifier(n_neighbors=3)
    resultado=validacao_cruzada_por_modelo(classifier, dfinicial=dfinicial, dfprocessado=dfprocessado)
    st.dataframe(resultado[:])
    st.markdown("K=7")
    classifier = KNeighborsClassifier(n_neighbors=7)
    resultado=validacao_cruzada_por_modelo(classifier, dfinicial=dfinicial, dfprocessado=dfprocessado)
    st.dataframe(resultado[:])
    st.markdown("K=10")
    classifier = KNeighborsClassifier(n_neighbors=10)
    resultado=validacao_cruzada_por_modelo(classifier, dfinicial=dfinicial, dfprocessado=dfprocessado)
    st.dataframe(resultado[:])


def naive_bayes(dfinicial, dfprocessado):
    st.markdown("""
                O Naive Bayes é um algoritmo de aprendizado de máquina que usa o Teorema de Bayes para calcular a probabilidade de uma instância pertencer a uma classe específica com base nas probabilidades das características condicionadas à classe. Embora faça a suposição simplificada de independência entre as características, o Naive Bayes é eficaz em tarefas de classificação, como filtragem de spam e categorização de texto, sendo particularmente útil em cenários com grandes conjuntos de dados e características categóricas, apesar de suas limitações.
                """, 
                unsafe_allow_html=True)
    classifier = GaussianNB()
    resultado=validacao_cruzada_por_modelo(classifier, dfinicial=dfinicial, dfprocessado=dfprocessado)
    st.dataframe(resultado[:])


def arvore(dfinicial, dfprocessado):
    st.markdown("""
                Decision Tree (Árvore de Decisão) é um algoritmo de aprendizado de máquina usado para classificação e regressão. Ele divide o conjunto de dados em subconjuntos com base nas características para tomar decisões em forma de árvore. Cada nó representa uma escolha de atributo e cada folha representa uma classe ou valor de saída. As árvores de decisão são interpretables e podem ser usadas em uma variedade de domínios.
                """, 
                unsafe_allow_html=True)
    classifier = DecisionTreeClassifier(random_state=42)
    resultado=validacao_cruzada_por_modelo(classifier, dfinicial=dfinicial, dfprocessado=dfprocessado)
    st.dataframe(resultado[:])


main()