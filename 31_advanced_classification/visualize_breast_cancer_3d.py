import pandas as pd
import plotly.express as px
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Load Data
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df['Diagnosis'] = df['target'].map({0: 'Malignant', 1: 'Benign'})

print("📊 Data Loaded. Shape:", df.shape)

# 2. Normalize (StandardScaler) - Essential for PCA
features = data.feature_names
x = df.loc[:, features].values
x = StandardScaler().fit_transform(x)

# 3. PCA Projection to 3D
pca = PCA(n_components=3)
components = pca.fit_transform(x)

# Create a DataFrame for the 3D data
total_var = pca.explained_variance_ratio_.sum() * 100
df_3d = pd.DataFrame(data = components, columns = ['PC1', 'PC2', 'PC3'])
df_3d['Diagnosis'] = df['Diagnosis']
# Add some feature info to hover data (e.g., Mean Radius)
df_3d['Mean Radius'] = df['mean radius']
df_3d['Mean Texture'] = df['mean texture']

print(f"✨ PCA Complete. Top 3 components explain {total_var:.2f}% of the variance.")

# 4. Create Interactive 3D Plot
fig = px.scatter_3d(
    df_3d, 
    x='PC1', 
    y='PC2', 
    z='PC3',
    color='Diagnosis',
    color_discrete_map={'Malignant': 'red', 'Benign': 'green'},
    symbol='Diagnosis',
    opacity=0.7,
    title=f'3D PCA Visualization of Breast Cancer Data (Explains {total_var:.1f}% Variance)',
    hover_data=['Mean Radius', 'Mean Texture'],
    labels={
        'PC1': f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
        'PC2': f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)',
        'PC3': f'Principal Component 3 ({pca.explained_variance_ratio_[2]*100:.1f}%)'
    }
)

# Customize layout
fig.update_layout(
    margin=dict(l=0, r=0, b=0, t=50),
    scene=dict(
        xaxis_title='PC1 (Size/Shape)',
        yaxis_title='PC2 (Texture/Irregularity)',
        zaxis_title='PC3 (Detail)'
    )
)

# 5. Save
output_file = 'breast_cancer_3d_visualization.html'
fig.write_html(output_file)
print(f"✅ Interactive 3D plot saved to: {output_file}")
print("   Open this file in your browser to rotate, zoom, and explore the data clouds!")
