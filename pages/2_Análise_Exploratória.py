import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport
import streamlit.components.v1 as components


st.set_page_config(
    page_title= "Profiling de Dados",
    page_icon= "📊",
    layout= "wide",
    initial_sidebar_state= "collapsed",
    menu_items= {
        'About': "Desenvolvido por:\n- Dayvson Moura;\n- Thales Mayrinck "
    }
)


@st.cache_data
def profile(df):
   report = ProfileReport(df, title="Análise Exploratória").to_html()
   components.html(report, height=1000, width=1120, scrolling=True)


def main():
    header()
    df = pd.read_csv("./data/covid_no_mexico.csv")
    profile(df)


def header():
    st.header("Análise Exploratória de Dados")
    st.markdown("""
                Esta página apresenta a análise exploratória de dados realizada a partir do <i>dataset</i>.
                """,
                unsafe_allow_html= True)

    
main()
