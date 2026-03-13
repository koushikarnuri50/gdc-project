import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Global Development Clustering", layout="wide")

st.title(" global development measurement project ")

# Sidebar
st.sidebar.header("Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel",
    type=["csv","xlsx"]
)

if uploaded_file:

    # Load dataset
    if uploaded_file.name.endswith("csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Dataset info
    st.subheader("Dataset Information")

    col1, col2 = st.columns(2)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])

    st.write("Column Types")
    st.write(df.dtypes)

    # Numeric columns
    df_num = df.select_dtypes(include="number")

    # Feature selection
    st.sidebar.header("Feature Selection")

    features = st.sidebar.multiselect(
        "Select Features",
        df_num.columns,
        default=df_num.columns
    )

    df_num = df_num[features]

    # Handle missing values
    df_num = df_num.fillna(df_num.median())

    # Scaling
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_num)

    # Feature distribution
    st.subheader("Feature Distribution")

    feature = st.selectbox("Select Feature", df_num.columns)

    fig = px.histogram(df, x=feature)
    st.plotly_chart(fig, use_container_width=True)

    # Model selection
    st.sidebar.header("Model Selection")

    model_name = st.sidebar.selectbox(
        "Choose Model",
        [
            "KMeans",
            "DBSCAN",
            "Hierarchical",
            "Mean Shift",
            "Gaussian Mixture",
            "KNN"
        ]
    )

    run = st.sidebar.button("Run Model")

    if run:

        if model_name == "KMeans":
            model = KMeans(n_clusters=3, random_state=42)
            labels = model.fit_predict(scaled_data)

        elif model_name == "DBSCAN":
            model = DBSCAN(eps=0.5)
            labels = model.fit_predict(scaled_data)

        elif model_name == "Hierarchical":
            model = AgglomerativeClustering(n_clusters=3)
            labels = model.fit_predict(scaled_data)

        elif model_name == "Mean Shift":
            model = MeanShift()
            labels = model.fit_predict(scaled_data)

        elif model_name == "Gaussian Mixture":
            model = GaussianMixture(n_components=3, random_state=42)
            labels = model.fit_predict(scaled_data)

        elif model_name == "KNN":

            kmeans = KMeans(n_clusters=3)
            cluster_labels = kmeans.fit_predict(scaled_data)

            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(scaled_data, cluster_labels)

            labels = knn.predict(scaled_data)

        df["Cluster"] = labels

        # Silhouette score
        st.subheader("Model Performance")

        if len(set(labels)) > 1:
            score = silhouette_score(scaled_data, labels)
            st.metric("Silhouette Score", round(score,3))

        # PCA 2D
        st.subheader("PCA Cluster Visualization (2D)")

        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(scaled_data)

        fig = px.scatter(
            x=pca_data[:,0],
            y=pca_data[:,1],
            color=labels.astype(str),
            title=f"{model_name} Clusters"
        )

        st.plotly_chart(fig, use_container_width=True)

        # PCA 3D
        st.subheader("3D Cluster Visualization")

        pca3 = PCA(n_components=3)
        pca3_data = pca3.fit_transform(scaled_data)

        fig3 = px.scatter_3d(
            x=pca3_data[:,0],
            y=pca3_data[:,1],
            z=pca3_data[:,2],
            color=labels.astype(str),
            title="3D Cluster Visualization"
        )

        st.plotly_chart(fig3, use_container_width=True)

        # Cluster summary
        st.subheader("Cluster Summary")

        cluster_summary = df.groupby("Cluster").mean(numeric_only=True)
        st.dataframe(cluster_summary)

        # Cluster distribution
        st.subheader("Cluster Distribution")

        cluster_counts = df["Cluster"].value_counts().reset_index()
        cluster_counts.columns = ["Cluster","Count"]

        fig2 = px.bar(
            cluster_counts,
            x="Cluster",
            y="Count",
            color="Cluster"
        )

        st.plotly_chart(fig2, use_container_width=True)

        # Determine cluster meaning
        cluster_scores = cluster_summary.mean(axis=1)

        best_cluster = cluster_scores.idxmax()
        worst_cluster = cluster_scores.idxmin()

        # Country cluster finder
        if "Country" in df.columns:

            st.subheader("Country Cluster Finder")

            country = st.selectbox(
                "Select Country",
                sorted(df["Country"].unique())
            )

            # Show only ONE row (average values)
            result = df[df["Country"] == country].groupby("Country").mean(numeric_only=True).reset_index()

            cluster_value = df[df["Country"] == country]["Cluster"].mode()[0]
            result["Cluster"] = cluster_value

            st.write("### Selected Country Details")
            st.dataframe(result[["Country","Cluster"]])

            if cluster_value == best_cluster:

                st.success(
                    f"{country} belongs to Cluster {cluster_value}, which represents the most developed countries with strong economic indicators."
                )

            elif cluster_value == worst_cluster:

                st.warning(
                    f"{country} belongs to Cluster {cluster_value}, which represents less developed countries with development challenges."
                )

            else:

                st.info(
                    f"{country} belongs to Cluster {cluster_value}, which represents developing countries with improving economic growth."
                )

        # Cluster meaning section
        st.subheader("Cluster Meaning")

        for cluster in cluster_summary.index:

            st.write(f"### Cluster {cluster}")

            if cluster == best_cluster:
                st.write("This cluster represents the most developed countries.")

            elif cluster == worst_cluster:
                st.write("This cluster represents less developed countries.")

            else:
                st.write("This cluster represents developing countries.")

            st.dataframe(cluster_summary.loc[[cluster]])

        # Best country analysis
        if "Country" in df.columns:

            st.subheader("Best Country Analysis")

            numeric_cols = df.select_dtypes(include="number").columns

            df["Development_Score"] = df[numeric_cols].mean(axis=1)

            best_country_row = df.loc[df["Development_Score"].idxmax()]

            best_country = best_country_row["Country"]
            best_cluster_country = best_country_row["Cluster"]

            st.success(f"Best Performing Country: {best_country}")

            st.write("Cluster:", best_cluster_country)

            st.write(
                f"""
                Based on the development indicators, **{best_country}**
                has the highest overall development score.

                It belongs to **Cluster {best_cluster_country}**, representing
                countries with strong economic performance and development indicators.
                """
            )

        # Download results
        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Cluster Results",
            csv,
            "cluster_results.csv",
            "text/csv"
        )