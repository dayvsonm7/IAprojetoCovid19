import streamlit as st
import pandas as pd

st.set_page_config(
    page_title= "Home",
    page_icon= "🏠",
    layout= "wide",
    initial_sidebar_state= "auto",
    menu_items= {
        'About': "Desenvolvido por:\n- Dayvson Moura;\n- Thales Mayrinck;  "
    }
)


@st.cache_data
def ler_dataset():
    return pd.read_csv("./data/mexico_covid19.csv").head(100)





def about_dataset():
    st.markdown("""
                ### Machine Learning no Diagnóstico de Covid-19.
                O trabalho aborda a tarefa crucial de prever se uma pessoa está infectada com COVID-19 ou não, utilizando técnicas de classificação binária. Nesse contexto, diversos algoritmos podem ser empregados para realizar essa categorização precisa. Uma abordagem sugerida é a utilização de algoritmos de Aprendizado de Máquina, como Support Vector Machines (SVM), Random Forest e Naive Bayes. Esses algoritmos têm demonstrado eficácia na análise de dados médicos e podem extrair padrões complexos a partir de características clínicas e resultados de testes.
                """,
                unsafe_allow_html= True)
    st.dataframe(ler_dataset())



about_dataset()
