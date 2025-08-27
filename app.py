# import streamlit as st
# import torch
# import torch.nn.functional as F
# import pandas as pd
# import numpy as np
# import json
# from collections import Counter
# from sklearn.neighbors import NearestNeighbors
# from sklearn.cluster import KMeans
# import time

# # import sys
# # from pathlib import Path
# # sys.path.append(str(Path(__file__).parent.parent))

# from src.constants import CLASS_EMBEDDINGS_PATH, RANDOM_STATE, DEVICE, CLEANED_GPC_PATH, PRODUCT_TEST_EMBEDDINGS_PATH, CLEANED_TEST_DATA_PATH

# from src.utils import load_embedding_model, cluster_topk_classes
# from src.modules.models import KMeansModels, KMeansModelConfig
# import teradataml
# import teradatasql
# from teradataml import DataFrame

# from src.modules.db import TeradataDatabase


# @st.cache_resource
# def get_db():
#     db = TeradataDatabase()
#     db.connect()     # sets teradataml context
#     return db

# # Configure Streamlit page
# st.set_page_config(
#     page_title="Product Classification System",
#     page_icon="🛍️",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS for better Arabic text support
# st.markdown("""
# <style>
# .arabic-text {
#     direction: rtl;
#     text-align: right;
#     font-family: 'Arial Unicode MS', 'Tahoma', sans-serif;
# }
# .english-text {
#     direction: ltr;
#     text-align: left;
# }
# .product-name {
#     max-width: 300px;
#     word-wrap: break-word;
# }
# </style>
# """, unsafe_allow_html=True)

# # @st.cache_data
# # def load_data():
# #     product_df = pd.read_csv(CLEANED_TEST_DATA_PATH)
# #     class_df = pd.read_csv(CLEANED_GPC_PATH)
# #     product_embedding_df = pd.read_csv(PRODUCT_TEST_EMBEDDINGS_PATH)
# #     class_embedding_df = pd.read_csv(CLASS_EMBEDDINGS_PATH)
    
# #     product_full = product_embedding_df.merge(product_df, on="id")
# #     class_full = class_embedding_df.merge(class_df, on="id")
    
# #     return product_full, class_full

# @st.cache_data
# def load_data():

#     db = get_db()
#     # teradataml DataFrames
#     t_products = DataFrame.from_table("products", schema_name="demo_user")[["id", "translated_name"]]
#     t_classes  = DataFrame.from_table("classes",  schema_name="demo_user")[["id", "class_name"]]
#     t_pemb     = DataFrame.from_table("p_embeddings", schema_name="demo_user")
#     t_cemb     = DataFrame.from_table("c_embeddings", schema_name="demo_user")

#     # -> pandas
#     products_df = t_products.to_pandas()
#     classes_df  = t_classes.to_pandas()
#     p_emb_df    = t_pemb.to_pandas()
#     c_emb_df    = t_cemb.to_pandas()

#     # Merge embeddings to names
#     product_full = p_emb_df.merge(products_df, on="id", how="left")
#     class_full   = c_emb_df.merge(classes_df,  on="id", how="left")

#     return product_full, class_full


# # @st.cache_data
# # def load_embeddings(_product_full, _class_full):

# #     products_embeddings = [json.loads(embedding) for embedding in _product_full["embeddings"].tolist()]
# #     products_embeddings = torch.tensor(products_embeddings, dtype=torch.float16, device=DEVICE)
    
# #     classes_embeddings = [json.loads(embedding) for embedding in _class_full["embeddings"].tolist()]
# #     classes_embeddings = torch.tensor(classes_embeddings, dtype=torch.float16, device=DEVICE)
    
# #     return products_embeddings, classes_embeddings

# @st.cache_data
# def load_embeddings(product_full: pd.DataFrame, class_full: pd.DataFrame):
#     # Find all embedding columns (embed_0, embed_1, ...)
#     prod_embed_cols = sorted([c for c in product_full.columns if c.startswith("embed_")],
#                              key=lambda x: int(x.split("_")[1]))
#     class_embed_cols = sorted([c for c in class_full.columns if c.startswith("embed_")],
#                               key=lambda x: int(x.split("_")[1]))

#     # Convert to numpy then to torch
#     prod_np = product_full[prod_embed_cols].to_numpy(dtype=np.float32, copy=False)
#     class_np = class_full[class_embed_cols].to_numpy(dtype=np.float32, copy=False)

#     products_embeddings = torch.tensor(prod_np, dtype=torch.float16, device=DEVICE)
#     classes_embeddings  = torch.tensor(class_np, dtype=torch.float16, device=DEVICE)

#     return products_embeddings, classes_embeddings


# def detect_language(text):
#     if not isinstance(text, str):
#         return "unknown"
#     arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
#     total_chars = len([c for c in text if c.isalpha()])
    
#     if total_chars == 0:
#         return "unknown"
    
#     arabic_ratio = arabic_chars / total_chars
#     return "arabic" if arabic_ratio > 0.3 else "english"

# class KMeansClassifier:
#     def __init__(self, products_embeddings, classes_embeddings, class_full, n_clusters=40, topk=3):
#         self.products_embeddings = products_embeddings
#         self.classes_embeddings = classes_embeddings
#         self.class_full = class_full
#         self.n_clusters = n_clusters
#         self.topk = topk
#         self.device = classes_embeddings.device

#         config = KMeansModelConfig(n_clusters=n_clusters, topk=topk)
#         self.kmeans_model = KMeansModels(config)
      
#         products_np = products_embeddings.cpu().numpy()
#         self.kmeans_model.fit(products_np)
        
#         self.centroid_classes = self.kmeans_model.get_centroid_classes(classes_embeddings)
        
#     def classify_single_product(self, product_embedding):
#         if torch.is_tensor(product_embedding):
#             product_np = product_embedding.cpu().numpy().reshape(1, -1)
#         else:
#             product_np = np.array(product_embedding).reshape(1, -1)
        
#         cluster_label = self.kmeans_model.kmeans.predict(product_np)[0]

#         cluster_classes = self.centroid_classes.get(cluster_label)
        
#         if cluster_classes is not None and len(cluster_classes) > 0:
    
#             if self.topk and len(cluster_classes[0]) > 1:
#                 classes_list = cluster_classes[0].cpu().tolist()
#                 counter = Counter(classes_list)
#                 pred_class_idx, count = counter.most_common(1)[0]
#                 confidence = count / len(classes_list)
#             else:
                
#                 pred_class_idx = cluster_classes[0].item() if torch.is_tensor(cluster_classes[0]) else cluster_classes[0][0]
#                 confidence = 1.0
            
#             pred_class_name = self.class_full["class_name"].iloc[pred_class_idx]
            
#             return {
#                 'predicted_class_idx': pred_class_idx,
#                 'predicted_class_name': pred_class_name,
#                 'confidence': confidence,
#                 'cluster': cluster_label,
#                 'method': 'K-Means'
#             }
#         else:
#             return {
#                 'predicted_class_idx': -1,
#                 'predicted_class_name': 'Unknown',
#                 'confidence': 0.0,
#                 'cluster': cluster_label,
#                 'method': 'K-Means'
#             }

# class KNNClassifier:
#     def __init__(self, class_embeddings, class_full, k=3):
#         self.class_embeddings = class_embeddings
#         self.class_full = class_full
#         self.k = k
#         self.device = class_embeddings.device
     
#         self.class_embeddings_norm = F.normalize(class_embeddings, p=2, dim=1)
        
#     def classify_single_product(self, product_embedding):
#         if not torch.is_tensor(product_embedding):
#             product_embedding = torch.tensor(product_embedding, dtype=torch.float16, device=self.device)
        
#         if product_embedding.dim() == 1:
#             product_embedding = product_embedding.unsqueeze(0)
            
#         product_embedding_norm = F.normalize(product_embedding, p=2, dim=1)
        
#         similarity_scores = torch.mm(product_embedding_norm, self.class_embeddings_norm.T)
        
#         top_similarities, top_indices = torch.topk(similarity_scores, k=self.k, dim=1)
        
#         top_indices_cpu = top_indices[0].cpu().numpy()
        
#         counter = Counter(top_indices_cpu)
#         pred_class_idx, count = counter.most_common(1)[0]
#         confidence = count / len(top_indices_cpu)
        
#         pred_class_name = self.class_full["class_name"].iloc[pred_class_idx]
        
#         return {
#             'predicted_class_idx': pred_class_idx,
#             'predicted_class_name': pred_class_name,
#             'confidence': confidence,
#             'top_classes': top_indices_cpu.tolist(),
#             'method': 'KNN'
#         }

# def display_product_name(name, max_length=50):
#     """Display product name with proper formatting for Arabic/English"""
#     if pd.isna(name) or name == '':
#         return "N/A"
    
#     name_str = str(name)
#     language = detect_language(name_str)
    
#     # Truncate if too long
#     display_name = name_str if len(name_str) <= max_length else f"{name_str[:max_length]}..."
    
#     if language == "arabic":
#         return f'<div class="arabic-text product-name">{display_name}</div>'
#     else:
#         return f'<div class="english-text product-name">{display_name}</div>'

# def main():
#     st.title("🛍️ Product Classification System")
#     st.markdown("### Dual Classification: K-Means + KNN using GPC Standard")
    
#     # Sidebar for configuration
#     st.sidebar.header("⚙️ Configuration")
    
#     # Load data
#     with st.spinner("Loading data..."):
#         product_full, class_full = load_data()
#         products_embeddings, classes_embeddings = load_embeddings(product_full, class_full)
    
#     st.sidebar.success(f"✅ Data loaded successfully!")
#     st.sidebar.info(f"📊 Products: {len(product_full):,}")
#     st.sidebar.info(f"🏷️ Classes: {len(class_full):,}")
#     st.sidebar.info(f"🖥️ Device: {DEVICE}")
    
#     n_clusters = 33
#     kmeans_topk = 3

#     k_neighbors = 3
    
#     # Initialize classifiers
#     if 'kmeans_classifier' not in st.session_state or st.session_state.get('current_clusters') != n_clusters:
#         with st.spinner("Initializing K-Means classifier..."):
#             st.session_state.kmeans_classifier = KMeansClassifier(
#                 products_embeddings, classes_embeddings, class_full, 
#                 n_clusters=n_clusters, topk=kmeans_topk
#             )
#             st.session_state.current_clusters = n_clusters
    
#     if 'knn_classifier' not in st.session_state or st.session_state.get('current_k') != k_neighbors:
#         st.session_state.knn_classifier = KNNClassifier(classes_embeddings, class_full, k=k_neighbors)
#         st.session_state.current_k = k_neighbors
    
#     # Main content area
#     st.header("📋 Product Data")
    
#     # Pagination controls
#     items_per_page = st.selectbox("Items per page", [5, 10, 25, 50], index=1)
#     total_items = len(product_full)
#     total_pages = (total_items - 1) // items_per_page + 1
    
#     col1, col2, col3 = st.columns([1, 2, 1])
#     with col2:
#         current_page = st.number_input(
#             f"Page (1 to {total_pages})", 
#             min_value=1, 
#             max_value=total_pages, 
#             value=1
#         )
    
#     # Calculate pagination indices
#     start_idx = (current_page - 1) * items_per_page
#     end_idx = min(start_idx + items_per_page, total_items)
    
#     # Display current page data
#     current_data = product_full.iloc[start_idx:end_idx].copy()
    
#     # Add classification results to session state if not exists
#     if 'classification_results' not in st.session_state:
#         st.session_state.classification_results = {}
    
#     # Display table with classify buttons
#     st.subheader(f"Showing items {start_idx + 1} to {end_idx} of {total_items}")
    
#     # Create columns for the table
#     col_id, col_name, col_actions, col_kmeans_result, col_knn_result, col_comparison = st.columns([1, 4, 2, 2, 2, 1])
    
#     with col_id:
#         st.write("**ID**")
#     with col_name:
#         st.write("**Product Name**")
#     with col_actions:
#         st.write("**Actions**")
#     with col_kmeans_result:
#         st.write("**K-Means Result**")
#     with col_knn_result:
#         st.write("**KNN Result**")
#     with col_comparison:
#         st.write("**Match**")
    
#     st.divider()
    
#     # Display each row with classify buttons
#     for idx, row in current_data.iterrows():
#         col_id, col_name, col_actions, col_kmeans_result, col_knn_result, col_comparison = st.columns([1, 4, 2, 2, 2, 1])
        
#         with col_id:
#             st.write(row['id'])
        
#         with col_name:
#             # Display product name with language support
#             product_name = row.get('cleaned_text', row.get('name', 'N/A'))
#             name_html = display_product_name(product_name, max_length=60)
#             st.markdown(name_html, unsafe_allow_html=True)
        
#         with col_actions:
#             # Two classify buttons side by side
#             col_kmeans_btn, col_knn_btn = st.columns(2)
            
#             with col_kmeans_btn:
#                 kmeans_key = f"kmeans_{row['id']}_{current_page}"
#                 if st.button("📊 K-Means", key=kmeans_key, use_container_width=True):
#                     # Get product embedding
#                     product_idx = product_full[product_full['id'] == row['id']].index[0]
#                     product_embedding = products_embeddings[product_idx]
                    
#                     # Classify with K-Means
#                     with st.spinner("K-Means..."):
#                         start_time = time.time()
#                         result = st.session_state.kmeans_classifier.classify_single_product(product_embedding)
#                         end_time = time.time()
#                         result['processing_time'] = end_time - start_time
                    
#                     # Store result
#                     if row['id'] not in st.session_state.classification_results:
#                         st.session_state.classification_results[row['id']] = {}
#                     st.session_state.classification_results[row['id']]['kmeans'] = result
#                     st.rerun()
            
#             with col_knn_btn:
#                 knn_key = f"knn_{row['id']}_{current_page}"
#                 if st.button("🔍 KNN", key=knn_key, use_container_width=True):
#                     # Get product embedding
#                     product_idx = product_full[product_full['id'] == row['id']].index[0]
#                     product_embedding = products_embeddings[product_idx]
                    
#                     # Classify with KNN
#                     with st.spinner("KNN..."):
#                         start_time = time.time()
#                         result = st.session_state.knn_classifier.classify_single_product(product_embedding)
#                         end_time = time.time()
#                         result['processing_time'] = end_time - start_time
                    
#                     # Store result
#                     if row['id'] not in st.session_state.classification_results:
#                         st.session_state.classification_results[row['id']] = {}
#                     st.session_state.classification_results[row['id']]['knn'] = result
#                     st.rerun()
        
#         # Display K-Means results
#         with col_kmeans_result:
#             if (row['id'] in st.session_state.classification_results and 
#                 'kmeans' in st.session_state.classification_results[row['id']]):
#                 result = st.session_state.classification_results[row['id']]['kmeans']
#                 class_name = result['predicted_class_name']
#                 confidence = result['confidence']
#                 color = "🟢" if confidence > 0.7 else "🟡" if confidence > 0.4 else "🔴"
                
#                 st.write(class_name)
#                 st.write(f"{color} {confidence:.2f}")
#             else:
#                 st.write("—")
        
#         # Display KNN results
#         with col_knn_result:
#             if (row['id'] in st.session_state.classification_results and 
#                 'knn' in st.session_state.classification_results[row['id']]):
#                 result = st.session_state.classification_results[row['id']]['knn']
#                 class_name = result['predicted_class_name']
#                 confidence = result['confidence']
#                 color = "🟢" if confidence > 0.7 else "🟡" if confidence > 0.4 else "🔴"
                
#                 st.write(class_name)
#                 st.write(f"{color} {confidence:.2f}")
#             else:
#                 st.write("—")
        
#         # Display comparison
#         with col_comparison:
#             if (row['id'] in st.session_state.classification_results and 
#                 'kmeans' in st.session_state.classification_results[row['id']] and
#                 'knn' in st.session_state.classification_results[row['id']]):
#                 kmeans_class = st.session_state.classification_results[row['id']]['kmeans']['predicted_class_idx']
#                 knn_class = st.session_state.classification_results[row['id']]['knn']['predicted_class_idx']
                
#                 if kmeans_class == knn_class:
#                     st.write("✅")
#                 else:
#                     st.write("❌")
#             else:
#                 st.write("—")
        
#         st.divider()
    
#     # Batch classification options
#     st.header("⚡ Batch Operations")
#     col1, col2, col3, col4 = st.columns(4)
    
#     with col1:
#         if st.button("📊 K-Means All", use_container_width=True):
#             progress_bar = st.progress(0)
#             status_text = st.empty()
            
#             for i, (idx, row) in enumerate(current_data.iterrows()):
#                 if (row['id'] not in st.session_state.classification_results or 
#                     'kmeans' not in st.session_state.classification_results[row['id']]):
#                     status_text.text(f"K-Means: {i+1}/{len(current_data)}")
                    
#                     # Get product embedding
#                     product_idx = product_full[product_full['id'] == row['id']].index[0]
#                     product_embedding = products_embeddings[product_idx]
                    
#                     # Classify
#                     result = st.session_state.kmeans_classifier.classify_single_product(product_embedding)
                    
#                     if row['id'] not in st.session_state.classification_results:
#                         st.session_state.classification_results[row['id']] = {}
#                     st.session_state.classification_results[row['id']]['kmeans'] = result
                
#                 progress_bar.progress((i + 1) / len(current_data))
            
#             status_text.text("✅ K-Means batch complete!")
#             time.sleep(1)
#             st.rerun()
    
#     with col2:
#         if st.button("🔍 KNN All", use_container_width=True):
#             progress_bar = st.progress(0)
#             status_text = st.empty()
            
#             for i, (idx, row) in enumerate(current_data.iterrows()):
#                 if (row['id'] not in st.session_state.classification_results or 
#                     'knn' not in st.session_state.classification_results[row['id']]):
#                     status_text.text(f"KNN: {i+1}/{len(current_data)}")
                    
#                     # Get product embedding
#                     product_idx = product_full[product_full['id'] == row['id']].index[0]
#                     product_embedding = products_embeddings[product_idx]
                    
#                     # Classify
#                     result = st.session_state.knn_classifier.classify_single_product(product_embedding)
                    
#                     if row['id'] not in st.session_state.classification_results:
#                         st.session_state.classification_results[row['id']] = {}
#                     st.session_state.classification_results[row['id']]['knn'] = result
                
#                 progress_bar.progress((i + 1) / len(current_data))
            
#             status_text.text("✅ KNN batch complete!")
#             time.sleep(1)
#             st.rerun()
    
#     with col3:
#         if st.button("🚀 Both Methods", use_container_width=True):
#             progress_bar = st.progress(0)
#             status_text = st.empty()
#             total_ops = len(current_data) * 2
#             op_count = 0
            
#             for i, (idx, row) in enumerate(current_data.iterrows()):
#                 product_idx = product_full[product_full['id'] == row['id']].index[0]
#                 product_embedding = products_embeddings[product_idx]
                
#                 if row['id'] not in st.session_state.classification_results:
#                     st.session_state.classification_results[row['id']] = {}
                
#                 # K-Means
#                 if 'kmeans' not in st.session_state.classification_results[row['id']]:
#                     status_text.text(f"K-Means: {i+1}/{len(current_data)}")
#                     result = st.session_state.kmeans_classifier.classify_single_product(product_embedding)
#                     st.session_state.classification_results[row['id']]['kmeans'] = result
                
#                 op_count += 1
#                 progress_bar.progress(op_count / total_ops)
                
#                 # KNN
#                 if 'knn' not in st.session_state.classification_results[row['id']]:
#                     status_text.text(f"KNN: {i+1}/{len(current_data)}")
#                     result = st.session_state.knn_classifier.classify_single_product(product_embedding)
#                     st.session_state.classification_results[row['id']]['knn'] = result
                
#                 op_count += 1
#                 progress_bar.progress(op_count / total_ops)
            
#             status_text.text("✅ Both methods complete!")
#             time.sleep(1)
#             st.rerun()
    
#     with col4:
#         if st.button("🗑️ Clear All", use_container_width=True):
#             st.session_state.classification_results = {}
#             st.rerun()
    
#     # Download results
#     if st.session_state.classification_results:
#         st.header("📥 Download Results")
        
#         # Prepare results for download
#         results_data = []
#         for product_id, results in st.session_state.classification_results.items():
#             row_data = {'product_id': product_id}
            
#             if 'kmeans' in results:
#                 row_data.update({
#                     'kmeans_class': results['kmeans']['predicted_class_name'],
#                     'kmeans_confidence': results['kmeans']['confidence']
#                 })
            
#             if 'knn' in results:
#                 row_data.update({
#                     'knn_class': results['knn']['predicted_class_name'],
#                     'knn_confidence': results['knn']['confidence']
#                 })
            
#             # Agreement check
#             if 'kmeans' in results and 'knn' in results:
#                 row_data['methods_agree'] = (results['kmeans']['predicted_class_idx'] == 
#                                            results['knn']['predicted_class_idx'])
            
#             results_data.append(row_data)
        
#         if results_data:
#             results_df = pd.DataFrame(results_data)
#             csv = results_df.to_csv(index=False)
            
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.download_button(
#                     "📥 Download All Results",
#                     csv,
#                     "classification_comparison.csv",
#                     "text/csv",
#                     use_container_width=True
#                 )
    
#     # Statistics
#     if st.session_state.classification_results:
#         st.header("📊 Classification Statistics")
        
#         # Calculate statistics
#         kmeans_results = []
#         knn_results = []
#         agreements = []
        
#         for results in st.session_state.classification_results.values():
#             if 'kmeans' in results:
#                 kmeans_results.append(results['kmeans'])
#             if 'knn' in results:
#                 knn_results.append(results['knn'])
            
#             if 'kmeans' in results and 'knn' in results:
#                 agree = results['kmeans']['predicted_class_idx'] == results['knn']['predicted_class_idx']
#                 agreements.append(agree)
        
#         # Display metrics
#         col1, col2, col3, col4 = st.columns(4)
        
#         with col1:
#             st.metric("K-Means Classifications", len(kmeans_results))
#             if kmeans_results:
#                 avg_conf = np.mean([r['confidence'] for r in kmeans_results])
#                 st.metric("K-Means Avg Confidence", f"{avg_conf:.3f}")
        
#         with col2:
#             st.metric("KNN Classifications", len(knn_results))
#             if knn_results:
#                 avg_conf = np.mean([r['confidence'] for r in knn_results])
#                 st.metric("KNN Avg Confidence", f"{avg_conf:.3f}")
        
#         with col3:
#             if agreements:
#                 agreement_rate = np.mean(agreements)
#                 st.metric("Method Agreement", f"{agreement_rate:.1%}")
#                 st.metric("Agreements", f"{sum(agreements)}/{len(agreements)}")
        
#         with col4:
#             if kmeans_results and knn_results:
#                 kmeans_times = [r.get('processing_time', 0) for r in kmeans_results]
#                 knn_times = [r.get('processing_time', 0) for r in knn_results]
#                 avg_kmeans_time = np.mean(kmeans_times) * 1000
#                 avg_knn_time = np.mean(knn_times) * 1000
#                 st.metric("K-Means Avg Time (ms)", f"{avg_kmeans_time:.1f}")
#                 st.metric("KNN Avg Time (ms)", f"{avg_knn_time:.1f}")

# if __name__ == "__main__":
#     main()


# # app_e5_only.py
# import streamlit as st
# import torch
# import torch.nn.functional as F
# import pandas as pd
# import numpy as np

# from teradataml import DataFrame
# from src.modules.db import TeradataDatabase
# from src.modules.models import SentenceEmbeddingModel, SentenceEmbeddingConfig

# # ---------- Config ----------
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# SCHEMA = "demo_user"           # change if needed
# PRODUCTS_TBL = "products"      # columns: id, translated_name (or cleaned_text)
# CLASSES_TBL  = "classes"       # columns: id, class_name
# P_EMB_TBL    = "p_embeddings"  # columns: id, embed_0..embed_N (optional)
# C_EMB_TBL    = "c_embeddings"  # columns: id, embed_0..embed_N (optional)

# # ---------- DB ----------
# @st.cache_resource
# def get_db():
#     db = TeradataDatabase()
#     db.connect()
#     return db

# @st.cache_data
# def load_data():
#     _ = get_db()  # ensure context set
#     t_products = DataFrame.from_table(PRODUCTS_TBL, schema_name=SCHEMA)[["id", "translated_name"]]
#     t_classes  = DataFrame.from_table(CLASSES_TBL,  schema_name=SCHEMA)[["id", "class_name"]]

#     # optional embeddings from DB (if present)
#     t_pemb = DataFrame.from_table(P_EMB_TBL, schema_name=SCHEMA)
#     t_cemb = DataFrame.from_table(C_EMB_TBL, schema_name=SCHEMA)

#     products_df = t_products.to_pandas()
#     classes_df  = t_classes.to_pandas()
#     p_emb_df    = t_pemb.to_pandas()
#     c_emb_df    = t_cemb.to_pandas()

#     product_full = p_emb_df.merge(products_df, on="id", how="left")
#     class_full   = c_emb_df.merge(classes_df,  on="id", how="left")
#     return product_full, class_full

# # ---------- E5 model ----------
# @st.cache_resource
# def load_encoder() -> SentenceEmbeddingModel:
#     cfg = SentenceEmbeddingConfig(
#         device=DEVICE,
#         dtype="float16",
#         model_id="intfloat/e5-large-v2",
#         truncate_dim=None,
#         convert_to_numpy=False,
#         convert_to_tensor=True,
#         use_prompt=True,
#         prompt_config={
#             # For E5, you can also split "query:" vs "passage:"; here we unify for simplicity
#             "classification": "passage: {text}"
#         },
#         model_kwargs={"torch_dtype": "float16"},
#     )
#     return SentenceEmbeddingModel(cfg)

# # ---------- Embeddings (two modes) ----------
# @st.cache_data(show_spinner=False)
# def load_embeddings_from_db(product_full: pd.DataFrame, class_full: pd.DataFrame):
#     prod_cols = sorted([c for c in product_full.columns if c.startswith("embed_")],
#                        key=lambda x: int(x.split("_")[1]))
#     cls_cols  = sorted([c for c in class_full.columns if c.startswith("embed_")],
#                        key=lambda x: int(x.split("_")[1]))
#     prod_np = product_full[prod_cols].to_numpy(dtype=np.float32, copy=False)
#     cls_np  = class_full[cls_cols].to_numpy(dtype=np.float32, copy=False)
#     prod = torch.tensor(prod_np, dtype=torch.float16, device=DEVICE)
#     cls  = torch.tensor(cls_np,  dtype=torch.float16, device=DEVICE)
#     # normalize for cosine
#     return F.normalize(prod.float(), p=2, dim=1).half(), F.normalize(cls.float(), p=2, dim=1).half()

# @st.cache_data(show_spinner=False)
# def load_embeddings_from_model(product_full: pd.DataFrame, class_full: pd.DataFrame):
#     enc = load_encoder()
#     prod_texts = product_full.get("translated_name", "").astype(str).tolist()
#     cls_texts  = class_full.get("class_name", "").astype(str).tolist()
#     prod = enc.get_embeddings(prod_texts, prompt_name="classification")
#     cls  = enc.get_embeddings(cls_texts,  prompt_name="classification")
#     if not torch.is_tensor(prod): prod = torch.tensor(prod)
#     if not torch.is_tensor(cls):  cls  = torch.tensor(cls)
#     prod = prod.to(DEVICE).to(torch.float16)
#     cls  = cls.to(DEVICE).to(torch.float16)
#     return F.normalize(prod.float(), p=2, dim=1).half(), F.normalize(cls.float(), p=2, dim=1).half()

# # ---------- Simple classifier using E5 similarity ----------
# def predict_topk(prod_vec: torch.Tensor, cls_mat: torch.Tensor, k: int = 3):
#     # prod_vec: [dim], cls_mat: [C, dim]  -> scores: [C]
#     if prod_vec.dim() == 1:
#         prod_vec = prod_vec.unsqueeze(0)
#     scores = torch.mm(prod_vec, cls_mat.T)  # cosine since normalized
#     vals, idx = torch.topk(scores, k=min(k, cls_mat.size(0)), dim=1)
#     return vals[0].detach().cpu().numpy(), idx[0].detach().cpu().numpy()

# # ---------- UI ----------
# st.set_page_config(page_title="Product Classification (E5 only)", page_icon="🧠", layout="wide")

# st.title("🧠 Product Classification — E5 only")

# with st.spinner("Loading data from Teradata..."):
#     product_full, class_full = load_data()
# st.success(f"Loaded: {len(product_full):,} products, {len(class_full):,} classes")

# use_model_now = st.sidebar.checkbox("Encode now with E5 (override DB embeddings)", value=False)

# with st.spinner("Preparing embeddings..."):
#     if use_model_now:
#         prod_emb, cls_emb = load_embeddings_from_model(product_full, class_full)
#     else:
#         prod_emb, cls_emb = load_embeddings_from_db(product_full, class_full)

# # Pagination + simple table
# items_per_page = st.selectbox("Items per page", [5, 10, 25, 50], index=1)
# total_items = len(product_full)
# total_pages = (total_items - 1)//items_per_page + 1
# current_page = st.number_input(f"Page (1..{total_pages})", 1, total_pages, 1)
# s, e = (current_page-1) * items_per_page, min(current_page * items_per_page, total_items)
# view = product_full.iloc[s:e][["id", "translated_name"]].reset_index(drop=True)

# st.subheader(f"Showing {s+1}–{e} / {total_items}")
# st.dataframe(view, hide_index=True, use_container_width=True)

# # Classify row id input + button
# col1, col2 = st.columns([2,1])
# with col1:
#     chosen_id = st.number_input("Enter product id to classify", min_value=int(product_full["id"].min()),
#                                 max_value=int(product_full["id"].max()), value=int(view.iloc[0]["id"]) if len(view) else 1)
# with col2:
#     if st.button("🔎 Classify with E5", use_container_width=True):
#         # locate row index
#         try:
#             pidx = product_full.index[product_full["id"] == chosen_id][0]
#         except IndexError:
#             st.error("ID not found in current dataset.")
#         else:
#             scores, idxs = predict_topk(prod_emb[pidx], cls_emb, k=3)
#             top1_idx = int(idxs[0])
#             top1_score = float(scores[0])

#             pred_name = class_full["class_name"].iloc[top1_idx]
#             st.success(f"Top-1: **{pred_name}** (score={top1_score:.3f})")
#             st.caption("Top‑3:")
#             for r, (sc, ci) in enumerate(zip(scores, idxs), start=1):
#                 st.write(f"{r}. {class_full['class_name'].iloc[int(ci)]} — {sc:.3f}")




# app_e5_only.py
import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from teradataml import DataFrame
from src.modules.db import TeradataDatabase

# ---------- Config ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCHEMA = "demo_user"
PRODUCTS_TBL = "products"
CLASSES_TBL  = "classes"
P_EMB_TBL    = "p_embeddings"
C_EMB_TBL    = "c_embeddings"
ACTUALS_TBL  = "actual_classes"

# ---------- DB helpers ----------
@st.cache_resource
def get_db():
    db = TeradataDatabase()
    db.connect()
    return db

@st.cache_data
def load_data():
    _ = get_db()  # ensure teradataml context

    # --- Base tables → pandas
    products_df = DataFrame.from_table(PRODUCTS_TBL, schema_name=SCHEMA)[["id", "translated_name"]].to_pandas()
    classes_df  = DataFrame.from_table(CLASSES_TBL,  schema_name=SCHEMA)[["id", "class_name"]].to_pandas()
    p_emb_df    = DataFrame.from_table(P_EMB_TBL,    schema_name=SCHEMA).to_pandas()
    c_emb_df    = DataFrame.from_table(C_EMB_TBL,    schema_name=SCHEMA).to_pandas()

    # Merge embeddings → names
    product_full = p_emb_df.merge(products_df, on="id", how="left")
    class_full   = c_emb_df.merge(classes_df,  on="id", how="left")

    # --- Ground truth (actual_classes) — be tolerant to schema differences
    try:
        actuals_df = DataFrame.from_table(ACTUALS_TBL, schema_name=SCHEMA).to_pandas()
        # normalize column names
        actuals_df.columns = [c.strip().lower() for c in actuals_df.columns]
        classes_df_norm = classes_df.rename(columns={"id": "class_id", "class_name": "class_name"})
        classes_df_norm.columns = [c.strip().lower() for c in classes_df_norm.columns]

        # figure product key
        prod_key = "product_id" if "product_id" in actuals_df.columns else ("id" if "id" in actuals_df.columns else None)
        if prod_key is None:
            st.warning(f"`{ACTUALS_TBL}` has no product key column (expected one of: product_id, id).")
            product_full["true_class_id"] = np.nan
            product_full["true_class_name"] = np.nan
            return product_full, class_full

        # figure class info: either class_id or class_name present
        if "class_id" in actuals_df.columns:
            # already numeric id → attach name
            gt = actuals_df[[prod_key, "class_id"]].rename(columns={prod_key: "id", "class_id": "true_class_id"})
            gt = gt.merge(classes_df_norm[["class_id", "class_name"]]
                          .rename(columns={"class_name": "true_class_name"}),
                          on="true_class_id", how="left")
        elif "class_name" in actuals_df.columns:
            # map name → id via classes table
            gt = actuals_df[[prod_key, "class_name"]].rename(columns={prod_key: "id"})
            gt = gt.merge(classes_df_norm, on="class_name", how="left")  # brings class_id
            gt = gt.rename(columns={"class_id": "true_class_id", "class_name": "true_class_name"})
            gt = gt[["id", "true_class_id", "true_class_name"]]
        else:
            st.warning(f"`{ACTUALS_TBL}` has no class column (expected one of: class_id, class_name).")
            product_full["true_class_id"] = np.nan
            product_full["true_class_name"] = np.nan
            return product_full, class_full

        # merge onto products
        before = len(product_full)
        product_full = product_full.merge(gt, on="id", how="left")
        matched = int(product_full["true_class_id"].notna().sum())
        st.caption(f"Ground-truth rows matched: {matched} / {before}")

    except Exception as e:
        st.warning(f"Could not read `{ACTUALS_TBL}`: {e}")
        product_full["true_class_id"] = np.nan
        product_full["true_class_name"] = np.nan

    return product_full, class_full

@st.cache_data
def load_embeddings_from_db(product_full, class_full):
    prod_cols = sorted([c for c in product_full.columns if c.startswith("embed_")],
                       key=lambda x: int(x.split("_")[1]))
    cls_cols  = sorted([c for c in class_full.columns if c.startswith("embed_")],
                       key=lambda x: int(x.split("_")[1]))
    if not prod_cols or not cls_cols:
        raise ValueError("Missing embed_* columns in DB tables.")
    prod_np = product_full[prod_cols].to_numpy(dtype=np.float32, copy=False)
    cls_np  = class_full[cls_cols].to_numpy(dtype=np.float32, copy=False)
    prod = torch.tensor(prod_np, dtype=torch.float16, device=DEVICE)
    cls  = torch.tensor(cls_np,  dtype=torch.float16, device=DEVICE)
    return F.normalize(prod.float(), p=2, dim=1).half(), F.normalize(cls.float(), p=2, dim=1).half()

def predict_topk(prod_vec, cls_mat, k=3):
    if prod_vec.dim() == 1:
        prod_vec = prod_vec.unsqueeze(0)
    scores = torch.mm(prod_vec, cls_mat.T)
    vals, idx = torch.topk(scores, k=min(k, cls_mat.size(0)), dim=1)
    return vals[0].cpu().numpy(), idx[0].cpu().numpy()

# ---------- UI ----------
st.title("🧠 Product Classification — E5 (DB embeddings + Ground Truth)")

with st.spinner("Loading data from Teradata…"):
    product_full, class_full = load_data()

with st.spinner("Preparing embeddings…"):
    prod_emb, cls_emb = load_embeddings_from_db(product_full, class_full)

# browse
items_per_page = st.selectbox("Items per page", [5, 10, 25, 50], 1)
total_items = len(product_full)
total_pages = (total_items - 1)//items_per_page + 1
page = st.number_input("Page", 1, total_pages, 1)
s, e = (page-1)*items_per_page, min(page*items_per_page, total_items)
st.dataframe(product_full.iloc[s:e][["id", "translated_name", "true_class_name"]], hide_index=True)

# classify
chosen_id = st.number_input("Enter product id", int(product_full["id"].min()), int(product_full["id"].max()), int(product_full.iloc[0]["id"]))
if st.button("🔎 Classify"):
    idx_match = product_full.index[product_full["id"] == chosen_id]
    if len(idx_match) == 0:
        st.error("ID not found.")
    else:
        pidx = int(idx_match[0])
        scores, idxs = predict_topk(prod_emb[pidx], cls_emb, k=3)
        top1_idx = int(idxs[0])
        pred_name = class_full["class_name"].iloc[top1_idx]
        pred_id   = int(class_full["id"].iloc[top1_idx])
        st.success(f"Prediction: {pred_name} (id={pred_id}, score={scores[0]:.3f})")

        true_id   = product_full.loc[pidx, "true_class_id"] if "true_class_id" in product_full else np.nan
        true_name = product_full.loc[pidx, "true_class_name"] if "true_class_name" in product_full else np.nan
        if pd.notna(true_id):
            st.write("✅ Correct" if int(true_id) == pred_id else "❌ Incorrect")
            st.write(f"Ground truth: {true_name} (id={int(true_id)})")