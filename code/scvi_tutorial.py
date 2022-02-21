
# coding: utf-8

# In[3]:


import scanpy as sc
import scvi
import numpy as np


# In[4]:


pbmc3k = scvi.data.read_h5ad("/home/fengling/Documents/nnbatch/data/tutorial/pbmc3k_raw.h5ad")


# In[5]:


pbmc5k = sc.read_10x_h5(
    "/home/fengling/Documents/nnbatch/data/tutorial/5k_pbmc_protein_v3_filtered_feature_bc_matrix.h5",
    gex_only=False
)


# In[6]:


pbmc5k.var_names_make_unique()


# In[7]:


pbmc5k.var.feature_types.astype("category").cat.categories


# In[8]:


scvi.data.organize_cite_seq_10x(pbmc5k)


# In[9]:


pbmc5k


# In[10]:


pbmc3k


# In[11]:


adata = pbmc5k.concatenate(pbmc3k)


# In[12]:


adata


# In[13]:


sc.pp.filter_genes(adata, min_counts=3)
sc.pp.filter_cells(adata, min_counts=3)


# In[14]:


adata.layers["counts"] = adata.X.copy()


# In[15]:


sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)


# In[16]:


adata.raw = adata


# In[17]:


scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key="batch")


# In[18]:


vae = scvi.model.SCVI(adata, n_layers=2, n_latent=30, gene_likelihood="nb")


# In[19]:


vae.train()


# In[ ]:


vae.save("/home/fengling/Documents/nnbatch/data/tutorial/vae_model_2/")

