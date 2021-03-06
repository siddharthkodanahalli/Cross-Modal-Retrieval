# Cross-Modal-Retrieval
This repository contains the code for all the phases of the Cross-Modal Retrieval project on Recipe1M dataset.

Cross-modal representation learning has become an interesting topic of research in the field of vision and language lately. Cross-modal representation learning aims to learn common representation from multiple views. In this work, we are mainly focusing on the cross-modal retrieval, wherein given a query in one view (e.g., image), we retrieve similar instances in another view (e.g., text). We have executed our work in three phases. 

In the first phase, we have implemented linear model like the Canonical Correlation Analysis(CCA) model to explore classical multi-view representation learning, where we discover a shared representation of observations from different views with the complex underlying correlation linearly. 

The next phase involves involves building upon the model implemented in first phase by using non-linear deep models like Deep Canonical Correlation Analysis(DCCA) and different loss functions to explore non-linear multi-view representation learning. 

In the final phase, we explored a transformer based model architecture which focuses on learning joint representations for textual and visual modalities by using transformer based encoders. This model lacked self-attention across multiple views. Thus, we have presented a new architecture using transformer encoder which uses self-attention across both image and text views. Both the models implemented in the final phase have been evaluated quantitatively and qualitatively in this project.
