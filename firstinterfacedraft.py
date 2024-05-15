from datetime import datetime, timedelta
import streamlit as st # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib as plt 
 
st.title('XAI Playground')
 
add_selectbox = st.sidebar.selectbox(
    "Dataset",
    ("dataset 1", "dataset 2", "dataset 3")
)

add_selectbox = st.sidebar.selectbox(
    "Model",
    ("Decision Tree", "Random Forest")
)

# Mindest- und Höchstdatum festlegen
min_date = datetime.now() - timedelta(days=365)
max_date = datetime.now()

# Datumsbereich auswählen
selected_date_range = st.sidebar.date_input('Select a date range:', 
                                        min_value=min_date, 
                                        max_value=max_date, 
                                        value=(min_date, max_date))

st.write('Selected date range:', selected_date_range)


def main():
    
    tabs = {
        'Dataset': tab1,
        'Model': tab2,
        'Feature influence': tab3
    }

    selected_tab = st.radio('Select Tab', list(tabs.keys()))

    tabs[selected_tab]()

def tab1():
    st.write('Content of dataset')
    with st.expander("Heatmap"):
        st.image("/Users/melina/AAI/heatmap.png", caption="heatmap")
    with st.expander("Correlation Matrix"):
        st.write('')
    with st.expander("Histogram"):
         st.write('')


    # Generiere einen zufälligen Datensatz
    np.random.seed(42)
    data = {
        'x': np.random.rand(100),
        'y': np.random.rand(100)
    }
    df = pd.DataFrame(data)

    # Titel für die App
    st.title('Zufälliger Datenplot mit Streamlit')

    # Zeige den Datensatz, wenn gewünscht
    if st.checkbox('Datensatz anzeigen'):
        st.write(df)

    # Wähle eine Visualisierungsmethode
    visualization_option = st.selectbox('Wähle eine Visualisierungsmethode:', ['Scatterplot', 'Histogramm'])


    # Visualisierung basierend auf der Auswahl erstellen
    if visualization_option == 'Scatterplot':
        plt.scatter(df['x'], df['y'], alpha=0.5)
        plt.xlabel('X')
        plt.ylabel('Y')
        st.plt.pyplot()

    elif visualization_option == 'Histogramm':
        plt.hist(df['x'], bins=20, color='skyblue', edgecolor='black')
        plt.xlabel('X')
        plt.ylabel('Häufigkeit')
        st.plt.pyplot()



def tab2():
    st.write('Content of model')

def tab3():
    st.write('Content of feature influence')

if __name__ == "__main__":
    main()