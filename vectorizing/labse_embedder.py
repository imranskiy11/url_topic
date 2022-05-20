from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np



class SentenceEmbedder:
    def __init__(self, device='cuda', path_enc_configs='LaBSE'):
        super(SentenceEmbedder, self).__init__()

        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(path_enc_configs)
        self.sentence_embedder = AutoModel.from_pretrained(path_enc_configs)
        self.sentence_embedder.to(self.device)



    def vectorize(self, word_list, max_length=256, truncation=True, padding=True, return_tensors='pt'):
        encoded_input = self.tokenizer(word_list,
                              padding=padding,
                              truncation=truncation,
                              max_length=max_length,
                              return_tensors=return_tensors).to(self.device)
        with torch.no_grad():
            model_output = self.sentence_embedder(**encoded_input)
        embeddings = model_output.pooler_output
        embeddings = torch.nn.functional.normalize(embeddings)
        if return_tensors == 'np':
            embeddings = embeddings.cpu().detach().numpy()
        return embeddings

    def word_embeddings_list(self, words: list, max_length=256, truncation=True, padding=True, return_tensors='pt'):
        lengths = list()
        for s in words:
            lengths.append(len(s))
        if len(lengths) == 0:
            median_len = 256
        else:
            median_len = int(np.median(np.array(lengths))) + 1

        if median_len >= 256:
            max_length = median_len
        params = {
            'max_length': max_length, 
            'truncation': truncation, 
            'padding': padding, 
            'return_tensors': return_tensors}
        return self.vectorize(words, **params)


    # def word_embedding(self, description: str, max_length=256, truncation=True, padding=True, return_tensors='pt'):
    #     params = {
    #         'max_length': max_length, 
    #         'truncation': truncation, 
    #         'padding': padding, 
    #         'return_tensors': return_tensors}
    #     return self.vectorize([description], **params)
