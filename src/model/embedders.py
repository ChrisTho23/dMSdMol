
# self.embed_tokens = BartScaledWordEmbedding(
#             config.vocab_size, embed_dim, self.padding_idx, embed_scale=embed_scale
#         )
import sys
sys.path.append('/Users/aaronfanous/Documents/EnvedaChallenge/dMSdMol2')
import fire
from torch.nn import Module
import torch.nn as nn
from src.data.load_data import load_and_split_parquet
from src.data.data import Mol2MSDataset

class Mol2MSModelEmbedding(Module):
    """_summary_

    Args:
        Module (_type_): _description_
    """
    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self,k,v)

        if self.config:
            for k,v in self.config.items():
                setattr(self,k,v)
        
        if self.originalEmbedding:
            # load weights here probablly 
            
            pass
        
        # nn.Embedding(categoricalDimension, embeddingDimension)
    def forward(self, x):
        """
        forward function for the embeddin


        Args:
            x (_type_): 
                


        """
        # 
        

      

        pass



# class ms2molEmbedding(Module):

#     def __init__(self):
#         pass
#     def forward(self,):
#         pass


def main():  # repo_names
    df=load_and_split_parquet("/Users/aaronfanous/Downloads/enveda_library_subset.parquet",0.1,0.1)
    unique_values= {}
    for split in df:
        # Convert to Pandas and get unique values for specific columns
        unique_values[split] = {
    "collision_energy": df[split].to_pandas()["collision_energy"].unique(),
    "instrument_type": df[split].to_pandas()["instrument_type"].unique()
    }

    # Print the unique values for each split
    for split, values in unique_values.items():
        print(f"Unique values in {split}:")
        print("Collision Energy:", values["collision_energy"])
        print("Instrument Type:", values["instrument_type"])
        print()


if __name__ == "__main__":
    fire.Fire(main)
