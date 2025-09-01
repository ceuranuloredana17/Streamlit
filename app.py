import streamlit as st
import pandas as pd
import altair as alt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

st.title("Analiză Ferme Lactate")

st.sidebar.header("Setări Aplicație")
uploaded_file = st.sidebar.file_uploader(
    "Încarcă Cleaned_Dairy_Dataset.csv", type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Preview Date")
    st.dataframe(df.head(10))

    locations = df['Location'].unique().tolist()
    selected_loc = st.sidebar.multiselect(
        "Selectează locații", locations, default=locations
    )
    df_filtered = df[df['Location'].isin(selected_loc)].copy()

    # Statistici descriptive
    st.subheader("Statistici descriptive")
    st.write(df_filtered.describe())

    # Venit mediu per locație
    st.subheader("Venit mediu (Approx. Total Revenue) per locație")
    rev_by_loc = (
        df_filtered.groupby('Location')['Approx. Total Revenue(INR)']
        .mean()
        .reset_index()
        .rename(columns={'Approx. Total Revenue(INR)': 'MeanRevenue'})
    )
    st.write(rev_by_loc)
    st.bar_chart(data=rev_by_loc, x='Location', y='MeanRevenue')

    # KPI HTML block
    total_ferme = len(df_filtered)
    avg_revenue = df_filtered['Approx. Total Revenue(INR)'].mean()
    max_cows = df_filtered['Number of Cows'].max()
    st.markdown(f"""
    <div style="background-color:#f0f4c3;padding:20px;border-radius:10px;">
      <h2 style="color:#33691e;text-align:center;">Ferme Lactate – KPI Overview</h2>
      <p style="font-size:16px;color:#558b2f;">
        <strong>Total ferme analizate:</strong> <span style="color:#c62828;">{total_ferme}</span><br>
        <strong>Venit mediu (INR):</strong> <span style="color:#2e7d32;">{avg_revenue:,.0f}</span><br>
        <strong>Max. număr vaci:</strong> <span style="color:#0277bd;">{max_cows}</span>
      </p>
    </div>
    """, unsafe_allow_html=True)

    # Clusterizare KMeans
    st.subheader("Clusterizare ferme după mărime și număr de vaci")
    features = df_filtered[['Total Land Area (acres)', 'Number of Cows']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    k = st.sidebar.slider("Număr de clustere", min_value=2, max_value=10, value=3)
    kmeans = KMeans(n_clusters=k, random_state=42)
    df_filtered['Cluster'] = kmeans.fit_predict(X_scaled)

    # Afișare rezultat clusterizare
    st.write(
        df_filtered[['Location', 'Number of Cows', 'Total Land Area (acres)', 'Cluster']]
        .head(10)
    )

    # Scatter plot clustere cu Altair
    st.subheader("Scatter plot clustere")
    scatter = alt.Chart(df_filtered).mark_circle(size=60).encode(
        x='Total Land Area (acres):Q',
        y='Number of Cows:Q',
        color=alt.Color('Cluster:N', legend=alt.Legend(title='Cluster'))
    ).interactive()
    st.altair_chart(scatter, use_container_width=True)


    
    # 3. SCATTER PLOTS PENTRU VARIABILELE PRINCIPALE - VERSIUNE STREAMLIT NATIVE
    st.write("\n**1. Relații vizuale cu venitul:**")

    # Pregătește datele pentru grafice
    if not df_filtered.empty:
        df_plot = df_filtered[['Number of Cows', 'Total Land Area (acres)', 'Approx. Total Revenue(INR)', 'Location']].dropna()
        
        # Limitează la 1000 de puncte pentru performanță
        if len(df_plot) > 1000:
            df_plot = df_plot.sample(n=1000, random_state=42)
            st.info(f"Afișez un eșantion de 1000 puncte din {len(df_filtered)} total pentru performanță.")
        
        st.write(f"Numărul de puncte afișate: {len(df_plot)}")
        
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Numărul de vaci vs Venit:**")
            
            # Folosește scatter chart nativ Streamlit
            try:
                chart_data_cows = df_plot[['Number of Cows', 'Approx. Total Revenue(INR)']].copy()
                chart_data_cows.columns = ['Numărul de Vaci', 'Venit (INR)']
                
                st.scatter_chart(
                    chart_data_cows,
                    x='Numărul de Vaci',
                    y='Venit (INR)',
                    height=300
                )
                
                # Afișează corelația
                corr_cows = df_plot['Number of Cows'].corr(df_plot['Approx. Total Revenue(INR)'])
                st.write(f"**Corelația**: {corr_cows:.3f}")
                
            except Exception as e:
                st.error(f"Eroare la graficul vaci vs venit: {e}")
                
                # Fallback simplu cu matplotlib prin st.pyplot
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.scatter(df_plot['Number of Cows'], df_plot['Approx. Total Revenue(INR)'], 
                          alpha=0.6, color='steelblue')
                ax.set_xlabel('Numărul de Vaci')
                ax.set_ylabel('Venit (INR)')
                ax.set_title('Numărul de vaci vs Venit')
                st.pyplot(fig)

        with col2:
            st.write("**Suprafața vs Venit:**")
            
            try:
                chart_data_land = df_plot[['Total Land Area (acres)', 'Approx. Total Revenue(INR)']].copy()
                chart_data_land.columns = ['Suprafața (acres)', 'Venit (INR)']
                
                st.scatter_chart(
                    chart_data_land,
                    x='Suprafața (acres)',
                    y='Venit (INR)',
                    height=300
                )
                
                # Afișează corelația
                corr_land = df_plot['Total Land Area (acres)'].corr(df_plot['Approx. Total Revenue(INR)'])
                st.write(f"**Corelația**: {corr_land:.3f}")
                
            except Exception as e:
                st.error(f"Eroare la graficul suprafață vs venit: {e}")
                
                # Fallback cu matplotlib
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.scatter(df_plot['Total Land Area (acres)'], df_plot['Approx. Total Revenue(INR)'], 
                          alpha=0.6, color='darkorange')
                ax.set_xlabel('Suprafața (acres)')
                ax.set_ylabel('Venit (INR)')
                ax.set_title('Suprafața vs Venit')
                st.pyplot(fig)

    # 4. ANALIZĂ PE LOCAȚII - VERSIUNE STREAMLIT NATIVE
    st.write("\n**2. Analiza pe locații:**")

    if 'Location' in df_filtered.columns:
        try:
            # Statistici pe locații
            st.write("**Distribuția veniturilor pe locații:**")
            
            location_stats = df_filtered.groupby('Location')['Approx. Total Revenue(INR)'].agg([
                'count', 'mean', 'median', 'std', 'min', 'max'
            ]).round(2)
            
            st.dataframe(location_stats)
            
            # Alternativ: histogramă pentru distribuția veniturilor
            st.write("**Histograma distribuției veniturilor:**")
            
            revenue_data = df_filtered['Approx. Total Revenue(INR)'].dropna()
            
            # Creează histograma cu matplotlib
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(revenue_data, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
            ax.set_xlabel('Venit (INR)')
            ax.set_ylabel('Frecvența')
            ax.set_title('Distribuția Veniturilor')
            ax.grid(True, alpha=0.3)
            
            # Adaugă linii pentru statistici descriptive
            mean_val = revenue_data.mean()
            median_val = revenue_data.median()
            ax.axvline(mean_val, color='red', linestyle='--', label=f'Media: {mean_val:,.0f}')
            ax.axvline(median_val, color='green', linestyle='--', label=f'Mediana: {median_val:,.0f}')
            ax.legend()
            
            st.pyplot(fig)
            
            # Box plot simplificat cu matplotlib
            st.write("**Box Plot pe primele 10 locații (după numărul de ferme):**")
            
            # Selectează top 10 locații după numărul de ferme
            top_locations = df_filtered['Location'].value_counts().head(10).index.tolist()
            df_box = df_filtered[df_filtered['Location'].isin(top_locations)]
            
            if len(df_box) > 0:
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Pregătește datele pentru box plot
                box_data = []
                labels = []
                for loc in top_locations:
                    loc_data = df_box[df_box['Location'] == loc]['Approx. Total Revenue(INR)'].dropna()
                    if len(loc_data) > 0:
                        box_data.append(loc_data)
                        labels.append(f"{loc}\n(n={len(loc_data)})")
                
                if box_data:
                    bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
                    
                    # Colorează box-urile
                    colors = plt.cm.Set3(range(len(box_data)))
                    for patch, color in zip(bp['boxes'], colors):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)
                    
                    ax.set_xlabel('Locația')
                    ax.set_ylabel('Venit (INR)')
                    ax.set_title('Distribuția Veniturilor pe Locații')
                    ax.tick_params(axis='x', rotation=45)
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                else:
                    st.warning("Nu s-au putut genera box plots - date insuficiente.")
            
        except Exception as e:
            st.error(f"Eroare la analiza pe locații: {e}")
            st.write("Detalii despre eroare pentru debugging:")
            st.write(str(e))

    # 5. VERIFICARE PENTRU VARIABILE CATEGORICE IMPORTANTE
    st.write("\n**3. Explorarea variabilelor categorice:**")

    categorical_columns = df_filtered.select_dtypes(include=['object', 'category']).columns.tolist()
    st.write(f"Variabile categorice: {categorical_columns}")

    for cat_col in categorical_columns[:3]:  # Limitează la primele 3 pentru a nu încărca interfața
        if cat_col != 'Location':  # Location deja analizată
            st.write(f"\n**Analiza pentru {cat_col}:**")
            
            unique_values = df_filtered[cat_col].nunique()
            st.write(f"Numărul de valori unice: {unique_values}")
            
            if unique_values <= 10:  # Doar pentru variabilele cu puține categorii
                cat_stats = df_filtered.groupby(cat_col)['Approx. Total Revenue(INR)'].agg([
                    'count', 'mean', 'std'
                ]).round(2)
                st.dataframe(cat_stats)

   

 

  
# 6. REGRESIE CU CELE MAI BUNE PREDICTORI
    st.write("\n**6. Regresie cu cei mai buni predictori:**")
    numeric_columns = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
    st.write(f"Variabile numerice găsite: {numeric_columns}")
    if len(numeric_columns) > 1:
        # Calculează corelațiile cu venitul
        correlations = df_filtered[numeric_columns].corr()['Approx. Total Revenue(INR)'].sort_values(ascending=False)
        
        st.write("**Corelații cu Approx. Total Revenue(INR):**")
        corr_df = pd.DataFrame({
            'Variabila': correlations.index,
            'Corelația': correlations.values,
            'Corelația_Absolută': abs(correlations.values)
        }).round(4)

    if len(numeric_columns) > 2:
        # Selectează primii 3-5 cei mai buni predictori (exclusiv venitul)
        best_predictors = correlations.drop('Approx. Total Revenue(INR)').abs().sort_values(ascending=False).head(5)
        
        st.write("**Cei mai buni predictori numerici:**")
        st.write(best_predictors)
        
        if best_predictors.iloc[0] > 0.1:  # Doar dacă există corelații decente
            # Încearcă regresie cu cei mai buni predictori
            predictor_cols = best_predictors.index.tolist()
            
            # Elimină valorile lipsă
            df_reg = df_filtered[predictor_cols + ['Approx. Total Revenue(INR)']].dropna()
            
            if len(df_reg) > 10:
                X = df_reg[predictor_cols]
                X = sm.add_constant(X)
                y = df_reg['Approx. Total Revenue(INR)']
                
                try:
                    model_improved = sm.OLS(y, X).fit()
                    
                    st.write(f"\n**Regresie îmbunătățită cu {len(predictor_cols)} predictori:**")
                    st.write(f"Predictori folosiți: {predictor_cols}")
                    st.text(str(model_improved.summary()))
                    
                    # Rezultate cheie
                    st.write(f"\n**Rezultate cheie:**")
                    st.write(f"• R²: {model_improved.rsquared:.4f} ({model_improved.rsquared*100:.2f}%)")
                    st.write(f"• R² ajustat: {model_improved.rsquared_adj:.4f}")
                    st.write(f"• F-statistic p-value: {model_improved.f_pvalue:.6f}")
                    
                except Exception as e:
                    st.error(f"Eroare la regresie îmbunătățită: {e}")
        else:
            st.warning("Nu există corelații suficient de puternice pentru o regresie validă.")

    # Informații suplimentare despre dataset
    st.write("\n**Informații suplimentare despre dataset:**")

    col_info1, col_info2 = st.columns(2)

    with col_info1:
        st.write("**Statistici rapide:**")
        st.write(f"• Total ferme: {len(df_filtered)}")
        st.write(f"• Locații unice: {df_filtered['Location'].nunique()}")
        st.write(f"• Venit mediu: {df_filtered['Approx. Total Revenue(INR)'].mean():,.0f} INR")
        st.write(f"• Venit median: {df_filtered['Approx. Total Revenue(INR)'].median():,.0f} INR")

    with col_info2:
        st.write("**Top 5 locații după numărul de ferme:**")
        top_5_locations = df_filtered['Location'].value_counts().head()
        for loc, count in top_5_locations.items():
            st.write(f"• {loc}: {count} ferme")

    
    st.write("\n**Sample din date (pentru verificare):**")
    sample_data = df_filtered[['Location', 'Number of Cows', 'Total Land Area (acres)', 'Approx. Total Revenue(INR)']].head()
    st.dataframe(sample_data)

else:
    st.info("Încarcă un fișier CSV pentru a începe analiza.")