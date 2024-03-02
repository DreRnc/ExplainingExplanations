# import pandas as pd
# import numpy as np

# def read_conllu(conll_file_path):

#     column_names = ["ID", "TOKEN", "LEMMA", "UPOS", "XPOS", \
#                     "FEATS", "HEAD", "DEPREL", "DEPS", "MISC"]
#     df = pd.read_csv(conll_file_path, delimiter='\t', comment='#', header=None, names=column_names)
#     df.reset_index(drop=True, inplace=True)

#     df['SAMPLE'] = None

#     sample = 0
#     for index, row in df.iterrows():
#         if(row["ID"]==1):
#             sample = sample+1
#         df.at[index, "SAMPLE"] = sample
    
#     return df

# def names_and_verbs(df, output_file_path):

#     modified_explanations = []
#     for i in range(1, np.max(df['SAMPLE']) + 1):
#         condition = (df['SAMPLE'] == i) & ((df['UPOS'] == 'NOUN') | (df['UPOS'] == 'VERB'))
#         df_i = df.loc[condition]
#         modified_explanations.append(' '.join(df_i["TOKEN"].values))

#     with open(output_file_path, "w") as f:
#         f.writelines(modified_explanations)

#     return

# def upos_tags(df, output_file_path):
#     modified_explanations = []
    
#     for i in range(1, np.max(df['SAMPLE']) + 1):
#         df_i = df.loc[df["SAMPLE"]==i]
#         modified_explanations.append(' '.join(df_i["UPOS"].values))

#     with open(output_file_path, "w") as f:
#         f.writelines(modified_explanations)

#     return


from conllu import parse

def distill_explanations(conllu_file, upos_tags, output_file):
    """
    Distill the explanations, keeping only terms of the categories specified in distilled_categories.
    
    Args:
        conllu_file (str): path to .conllu file containing the explanations
        upos_tags (lst of str): list of upos tags of syntactical categories to keep
        output_file (str): path to file for output where distilled explanations will be saved
        
    Returns:
        None
    """
    with open(conllu_file, "r", encoding="utf-8") as f:
        data = f.read()
        sentences = parse(data)

    with open(output_file, "w", encoding="utf-8") as f_out:
        for sentence in sentences:
            words = []
            for token in sentence:
                if token["upos"] in upos_tags:
                    words.append(token["form"])
            if words:
                f_out.write(" ".join(words) + "\n")