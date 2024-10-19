#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import codecs 
from rdkit import Chem
from SmilesPE.learner import * 
from SmilesPE.tokenizer import *
import numpy as np 
import pandas as pd
import random

def tokenize_deepsmiles(deepsmiles):
    deepsmiles_tokenized = [] 
    deepsmiles_tokenized_label = [] 
    
    spe_vocab = codecs.open("./deepSMILES_tokens_all_512.txt")
    spe = SPE_Tokenizer(spe_vocab)
    array_of_tokens = spe.tokenize(deepsmiles).split(" ")

    # need to change this every time we change token dict size 
    deepsmiles_tk = ['CC)', 'o', '%49', 'NCCO', '))))))))))))))', 'CNC=O)', 'Ccc', '%38', 'n6', 'CO))', '[As]', 'CCC)C)))', 'ccccCl)', 'CCCCN', 'c6))))))', 'cc6))))))))', 'COcccccc6', '[P]', 'CCC6', 'Ccn', 'n9', 'CC6))))))', '[N-]', 'Ncccc', '))))))', '%41', 'C=O)N', 'nc', 'cc', '%23', '[N+]=O)[O-]))', 'NC=O)', '))))))))))', 'c6', 'cO)', 'c%10', 'CCCCCCCC', ')))))', '%46', 'cccccc6', 'CC)C', 'CCNCC', '%50', 'CCC', '))))))))', 'COC=O)', '=O)))', '%15', 'CC=O)NCC', '[n+]', 'CF)F)F))', 'C=O)', 'nn', '[n-]', '[SH]', 'OCC', '%14', '-cccc', 'N=C', 'c6c%10', '[N+]', 'NC=', '%32', 'Ncn', '[N+]=O)[O-]', 'F)F)F', '=O', 'CC', 'B', '=C', 'P', 'cn', '))))))))))))', '4', '))))))))))))))))', 'OC))', ')))', 'cccccc', '%31', '%36', '[Cr]', '%26', '[Si]', '6O', 'Cl)c6', 'CCN', 'o5', 'F)F)', '[Se]', '6', 'COcccc', 'N)=O', 'cncc', '%37', '%30', ')', 'NCC', 'CO5', ')))))))))))))))', 'cccccc6)))))))', 'cccccc6))))))', 'cc6)))))))', '%20', 's', 'ccccF)', 'Ccccccc6)))))))', 'C#N', 'C=O)O))', 'c=O)', 'cccc6', 'F', '=O)', 'CCOC=O)', '%42', 'Cccccc', 'c', 'C)C', 'C)O', ')))))))))', 'CS', 'CCC)', 'O', 'CC)C)C', ')))))))))cc6', '%43', 'ccccc', 'cc6', '%44', '%11', 'C)', 'S=O)=O)', '5', 'n5', '[2H]', 'S=O)', 'C5', '%10', 'N)', '-cn', '))))', 'c6)))))))', '%27', '%34', 'nn5', 'NC', 'F)', '-cccccc6))))))', 'I', 'CN', 'nc5', '#', 'CNCC', 'CC5', '-cccccc6', 'Ccccccc6', 'CNC)', 'CCC)C', 'n', 'CC6', 'cccccc6))))))))', '[Al]', '%16', 'CN)=O', '%45', 's5', 'ncc', '[Na]', '[Co]', '=', '[Na+]', 'ccn', '[P+]', 'Cl)', '[s+]', '))', 'N', '%29', 'CC=O)', 'COccc', '8', 'CCO', 'O=C', '%18', '%24', '))))))))))cc6', 'cccn', 'CC6)))))))', 'CCCC', 'C=O)NCC', 'Cl', 'ccc', 'C=O)O', 'c5', ')))))))))))', 'Cccc', '%40', '[o+]', '7', '%28', '%47', 'C=C', '[PH]', 'cccccc96', 'cn6', '%35', 'S=O)=O)cccc', '[se]', 'N)N', 'C', 'OC=O)', 'N)N))))))', 'CCCN', '%106', 'CC=O)N', 'O=CO)', 'NC=O)cccc', 'c6))))))))', '96', ')))))))))))))', '%25', '[Fe]', '%12', '-ccc', 'O)', '%17', 'C6', 'S', 'cn5', 'Br', '%22', '[nH]', 'Ccccc', 'Br)', '[O]', 'CC=O)O', 'cnc', 'cccc', 'cc6))))))', '[S+]', 'ccc6', 'OC', '-cc', 'CO', 'C#', '[37Cl]', 'nc6', '%19', '[O-]', '[C-]', '*', '9', 'CF)F)F', '%13', '=N', '%21', 'cC)', '-', 'Ncccccc6', '=O)[O-]', '%33', 'S)', 'O=', 'cccccc6)', 'cOC))', '3', ')))))))', 'c9', 'CCNC=O)', 'C=']
    # print(f"size of {len(deepsmiles_tk)}")
        
    map_deepsmiles_token_pair = {deepsmiles: idx + 3 for idx, deepsmiles in enumerate(deepsmiles_tk)} # considers '0' , '1', and '2' tokens
    # make sure its rounded up first 
    deepsmiles_tokenized.append(0) # this is the beginning of sequence token 
    
    for e in array_of_tokens: 
        #expect float input
        if len(deepsmiles_tokenized) >= 256: 
            # input_list wont need eos token, only label_list
            deepsmiles_tokenized_label = deepsmiles_tokenized.copy() 
            deepsmiles_tokenized_label.pop(0) # remove <bos> token and add the eos token
            deepsmiles_tokenized_label.append(2)
            break
        deepsmiles_tokenized.append(map_deepsmiles_token_pair[e])

    if (256 - len(deepsmiles_tokenized)) > 0: # add padding
        deepsmiles_tokenized_label = deepsmiles_tokenized.copy() 
        deepsmiles_tokenized_label.pop(0) # remove bos token
        deepsmiles_tokenized_label.append(2) # before adding padding, firs add EOS token 
        
        for i in range(256 - len(deepsmiles_tokenized)):
            deepsmiles_tokenized_label.append(1) 
            deepsmiles_tokenized.append(1)
    # print(len(deepsmiles_tokenized))
    # print(len(deepsmiles_tokenized_label))

    return deepsmiles_tokenized, deepsmiles_tokenized_label

def untokenized_deepsmiles(deepsmiles_ids_tuple): 

    result_deepsmiles_tokens = [] 
    
    batch_size = deepsmiles_ids_tuple.size(0)
    
    

    deepsmiles_tk = ['CNCCOCC6)))))))', 'nc-cccccc6))))))', 'cc96)))))))))', '[37Cl]', 'CcccCl)', ')))))))))))))))))))))))', 'cccO)', 'CcccC)cc', 'Cco', 'c%13', 'Cnnc', 'oc5', 'cc5c9', '5=O', 'Cnccnc5', 'NC)C))', 'cccOC))', 'F)F)F))', 'CCNC6', 'CC=O)OC', 'cccco5', 'NCccn', 'C7=O', '[nH]5))))))', 'cccccc6%10', 'nccC)', 'c=O)nC)', 'NSC)=O)=O)))', 'ccccO)c', 's9))))))))))', 'NCCNC=O)', '%11', 'CCN', 'COcccc', 'Cl)cccc6', 'COcccc-cccc', 'o9)))))))))', 'cncO)', 'CCNC=O)cccc', '))))))', 'NC=O)OC', 'CcccC)', 'C=O)NCC', 'CCCCCC6)C8)))C6', 'SN)=O)=O', 'c%10', 'CCcccccc6', 'C=O)OCC)C)C', 'N)', 'CC)C)', 'Ccccccc6)OCO5', 'sc95', 'cnnc', 'n', 'NCCCC5)))))', '=O', 'F)))))))))', 'CF)F)F))', 'N#Ccccc', '%24', 'cnc5', '[nH]c', 'CC=O)Ncccccc6', 'OC))cc6))))))', 'c[N+]=O)[O-]))', 'ccccC)cc6', 'C6))))))))', '-cccC)', 'CCC3))))', '-cccccF)', '))))))))))))))))))cc6', 'O))', ')))))))))))))))c6', 'cc[nH]', 'O6)))))))', 'Cnn', 'F)F)F))))', 'cncc6', 'OC)))', 'c6=O', '-ccccF)cc6', 'CCNcccc', 'N)N))))))', 'CccF)cccc6', '-ccsc', 'oc=O)', '=CC=O)', 'OC))cc6)))))))', 'cc-cccccc6))))))', ')))))))c6', 'C=O)O', 'NC=NCC', 'CCCOcccc', 'Cccc=O)', 'Cccccnc6', 'cccccc%106', 'CCC=O)O', 'CCCCC))', 'O)cc6))))))', 'F)c', 'NC=O)OCC)C)C', 'ccc95', 'CCC)C)C6', ')))))))CCCC6', 's9', 'C=C', 'NC6=O', 'cccccc6))))))))', 'F)F)F)))', 'NNC=O)', '%40', 'CCCCCCCC6', 'Cl)c6))))))', 'Occcc', 'CCN)))', '-9', 'CCOC=O)cccc', 'c=O)n%10', 'Ccccccc6))))))', 'N5', '%135', 'CCCCC5)))))', 'ccccOC))cc6', 'ccccOC))cOC))', 'O=CNcccc', 'ncc', 'C=O)C', 's5)))))', 'CCC=O)O))))', 'Ccccccccc6', 'CC6)))))))c6', 'Ccccc-cccc', 'C4)))))', 'N=CN)NCC', 'NC=O)cccccc6))))))))', 'Ccc[nH]', ')))))))))))C6', 'C=N', ')))))))))CC6', 'COCCNCC', 'NCccccCl)', 'OCC)=O', 'Cl))))', 'Ccc[nH]cccccc96', 'O=[N+][O-])cccc', 'cNCC', 'CCC)C)cccc', 'CCC)C)C', 'CCCcccc', '5C=O)', 'P=O)O)O)))', 'n5c9', 'CF)F)F))cc6))))))))', 'Ccnc', '[Fe]', 'nc6%10', 'cccc6', 'OCCNCC', 'Ncccc', '-cncccccc6', 'CC=O)Nccccc', 'NC=O)COC=O)', 'Cccn', 'c6))))))))))', 'nc', 'Ccno', 'ccnn', 'CCN)=O', 'O=', 'CNC=O)NCC', 'Br)cc6', 'nc6)))))))', '*', 'O)cO)', 'CC%10', 'OCcccc', '[nH]9', '%44', 'CC=O)NC)', 'CCN5', '%10', '[nH]ccc5c9', 'OC))c6', 'cnncn5', 'CCCCCC', 'NC=N)', 'O=CNcccccc6', 'n5C', 'N)=O)=O', 'cn6))))))CC6', 'NC=O)ccn', 'CCCCCCC6)))))))', '%22', 'o5', 'C[n+]', 'cccnn5', 'nccc=O)', 'NC=N)N', 'cccccc6O', 'NCCNC)CC6))))))', 'CCCCC)', 'ccc6', 'cnoc', '-cccco5)))))', 'CC=N', '%26', 'Cncncc', '[S+][O-])', 'nc6))))))', '[nH]c-cccc', 'cccccCl)', '[N+][O-])', 'O=CNC=O)', 'cOC))c6))))))', '))))))))))n5', 'cccccc6n9', 'S)', 'CCC)C', 'Cccccccccc6c%10)))))))))))', 'N)=O', 'c6C#N', 'COcccccc6)', 'CC=O)NCC', 'CF)F', 'NC=O)NCC', 'Br)cc6))))))', 'OCCO6', '=N)', ')))C6', 'ccO)c6', 'n6', 'C5', '-ccccF)cc6))))))', 'CC))', 'ccC=O)O))', 'Ccccccc6', 'N)cc6', 'cs', 'Cl)', 'CF)', 'cncccccc6[nH]9', 'CC=O)O)))', 'C)O', 'OC5=O', 'Cl))))))))', 'NC=O)', 'cccccccc%106', 'COC))', 'CScccc', 'NC)C', 'COccccCNC=O)', 'F)cc6', 'c=O)c', 'n8', 'c6O', '%41', 'nn5))))))', 'O=CC', '%20', 'c6))))))cc6', 'NCCOCC6))))))', 'C=O)Ncccc', 'cCl)c6))))))))', 'C=O)O)))', 'cC=O)NCC', 'CCcccccc6)))))))', 'COcccc-cnc', 'CCC3)))', '-cnnn', 'cc%10', 'NCC', 'O=S=O)', 'cC#N))', 'nccc', 'CCCNC=O)cccc', '[nH]c95', '[n+]6', 'ccC)c6', '=NO5', 'C=O)OC)))', 'NO', 'C7', 'nc5', 'C=O)NCCC3', '[nH]5', 'CCNCCCC5', '-ncc', 'cOC))c6', 'COccccCNCC', 'Ccccccc6)', 'cc5', 'CCCNCC', '))))))))))))))))))))))))', 'ccsc', 'C%13', 'C)CC', 'NSC)=O)=O', 'o5)))))))', 'cc[nH]cccccc96))))))))))', 'S=O)=O)O', 'cc6)))))))', 'coc', 'CC6))))))n6', '6', 'C=O)N6', 'c6', 'COccccCl)cc6', 'ncccccc69', 'OCCCO', 'C=O)', 'C=O)NC)', 'Ncncccc', '-', 'cccccc69))))))))))', 'Cccsc', 'CCNC=O)', 'O=Ccccc', 'C)cc6)))))))', 'CCC)C)', 'Cl)))', 'cnc-cccc', 'CN', 'CCCCCCCC', 'Ncncnc', 'ccnc6', 'C=O)ccccc', 'CcccccC)c6', 'ccc6Cl', 'C%10=O', 'cccccBr)', 'C)C=', 'ccc%106', 'NC5=O', '-ccC)', 'ccF)cccc6', 'CNCCNC)', 'NC=O)cccco5', 'cccccc6)))))))))))', 'ccccF)cc6', 'NC=O)cccccc6', 'CN)=O))c', 'NC=N', 'C=O)O))))))', '=O)n', '-cccccF)c6))))))', 'c6))))))))', 'ccccCl)cc6Cl', ')))))))))))))))', 'cc[nH]cn5', 'COcccn', 'CCNcnc', 'C=O)OC', 'CCcccc', '%31', 'ccc[N+]=O)[O-]))', '))CC', 'cccncc6))))))))', 'CCC)cccc', 'cccn', '#', 'OCccn', 'CC=O)', 'ccccc-cccc', 'CC)C)))))))))))', 'CCOccccc', 'C)C', 'C=CCCCC6', 'CccccF)cc6', 'o5)))))cc6', 'CCC6', 'CNS=O)=O)cccc', 'Ncnnc', 'cn6)', 'NCCOCC6))))))))', 'Cl)))))))))))', 'CC)C', 'CO))))', 'ccccF)c', 'cc6', 'ccccCl)cc96', 'N=CN)N', 'COccccS=O)=O)NCC', 'ncS', '))))))))))', 'CCO)', 'ccccCl)cc6))))))))', 'Clcccc', '9))))))))))', 'CCCCN', 'ccccnc6', 'CNCCCC5', '[Si]', 'ccnc', 'ncncc=O)[nH]cN)nc6', 'CC6=O', '=CO)', 'Cl)))))))))', 'ncn', 'C=CCCC', 'NS=O)=O)cccc', 'CC)C))', 'Ncccccc6))))))', 'NC=O)cnc', 'CC6))))))))))', 'NC=O)Ncccccc6', 'CCC)O', 'OCC)=O)))', 's5)))))))))', 'sc5', 'Ccccc', '-ccncN)', '-ccnc', ')C8', 'CCCC', 'ccc-cccc', '))))))c6', 'cccC)ccc6', 'CCCC6', 'ccccCl)cCl)c6', 'Ncccccc6)', 'CC#N))', '[nH]n5', 'n%13', 'C5)))))))', 'cc95', 'CC3)))))', 'nn5)))))', 'cC)n5', 'cc6))))))))))))))', '%10)))))))))))', 'NCCCO', '%21', 'CN=C', 'CCC)=', 'CO6', '3', 'CNCCNCC', 'cc6)))))))))))))))', 'cco', 'c%106', 'CCCCCNC=O)', 'cccc6c%10', 'F', 'O=CCCC', '-cnn[nH]n5', 'cccF)', '[N+]=', 'C=O)cccccc6', 'CNC', 'ccs', 'C)C))', 'nccc=O)[nH]c6=O', 'SN)=O)=O))', 'CccccC)c', '[Al]', 'ccccccccc6c%10', 'ccBr)', 'C=O)cccccc6C%10=O', 'CC)C)C))))', 'O=CO', '[N+]=O)[O-]))c6))))))', 'cC)c5', 'ccccCl)cc6)))))))))', 'ccN)', 'ccC)cccc6', 'CNC)S=O)=O)', ')))cc6', '-ccccCl)cc6', 'n9)))))))))', 'NCCN5', 'NcccC)', '%13', 'CCCCCCCCCCCCCCCC', 'CCnc', 'CN)=O', 'ncnc', 'S=O)=O)cccc', '[O-]))', ')))))))))))CC)C', 'co', 'Ccc', 'CO)', '9', '[Na+]', ')))))))))))cccccc6', 'SN)=O)=O))cc6', ')))))))))))))c6', 'cC)c6', 'ccCl)cccc6', 'CcccccCl)c6', 'CccCl)cccc6', 'CCCS=O)=O)', 'C=O)NCCN)=O', 'OCCCC3', 'Nccn', '%30', 'CCC#N))', ')C8)))C6', 'nccn', 'Cncc', 'C=O)O))', '%14%10', 'ccccOC))', '))))))C6', '-cncc', 'cccOC))cOC))cOC))c6', 'cc6)))))))CC6', 'c=O)[nH]c6=O)))))))', 'cc6))))))))', 'ccccNC=O)', 'C5C)C', '))))))))))))))))cc6', '[nH]c=O)', 'cccccc6Cl', 'nnc', '))))))))c6', 'C=O)CCCC', 'P=O)O)O', 'Ncncc', 'ccccCl)', 'n6)))))))', 'CCCNC=', 'nn5)))))))', 'CCN6', 'CCncc', 's5)', 'CC)=O))', 'CNCCCC5))))))', 'COcccccc6', 'cO)c', '4', 'ccccCl)cc6))))))', 'COccccOC))c', 'NC=', 'N6', 'C5))))))cc6', 'c[N+]=O)[O-]))c6', 'O)))', 'C)cc6))))))', '=O)', '[N+]', 'NC=S)', 'ccc69', '#N))', 'N5C=O)', 'N)nc6', 'CO)))))', 'ccn5', 'Cnc', '-cc[nH]', 'cnnn', 's5)))))))', 'F))))))))))', 'CScnnc', 'CNC)C=O)', 'PO)', 'OC6', 'CcccF)', 'CCC)Occcc', 'OCC)', 'c69', 'CCNCCO', 'Cncncn5))))))', 'ccOC))c6O', 'O=CNccc', 'cF)c6)))))))', 'ccccCF)F)F))cc6', 'CNCcccccc6)))))))', 'ccc6F', '-cccccc6))))))nc5', 'C=O)OCC))))', 'C4', 'ccNCC', 'CcccO)', 'NC6', '[N+]C)', 'Occccc', 'C6=O', 'c9', 'cn6))))))))', '[O]', '[nH]cccc', 'CccccC=O)N', '-ccc', 'cccF)c', 'COcccccc6))))))))', 'ccc6O', '[N+]=O)[O-]))c6', 'C=S)', 'cs5)))))', 'cccccc96)))))))))', 'ccO)cccc6', 'Ccnoc', 'ccO)cc6', 'CccO)', ')CC8C6', 'C%12', 'ccccBr)cc6', '[Cr]', 'CNCC', ')))))))))))cc6)))))))', 'cccNC=O)', 'NC=O)CCCC', 'S=O)', 'cn6c9', 'C#N))cc6))))))', 'c6cc%10OC', 'C)C))))))', 'CCCO)', 'NCccccF)', 'NC=O)N', 'CCCN)=O', 'I)', 'CCCN=C', '=', 'cOC))cc6', 'ccccc6c%10)))))))))))', 'CC=C', ')))))))))))cc6OC', 'COccccNC=O)', 'cccCl)ccCl)', 'CcnnC)', 'CC6)))))))))))', 'c6))))))', 'NC)', ')))))))))))cc6', 'ncc6', 'cccccn6', 'C', 'cc-cccc', 'nccccc', '[n+][O-])', 'Br)cc6)))))))', ')))))))))))))))cc6', 'OCF)F)F)))', 'CC)C)C))', 'CCC6)))))))', 'CccccCl)cc6)))))))', 'C=C6', 'cccCl)ccc6', 'CC6))))))c6', 'CCCCCC6', '%47', 'c5n9', 'CC)C))cc6', 'CCCCCCCCCCCC', 'cccccc6)OCCO6', 'NccccF)cc6', 'CCCcccccc6', '=O)=O', 'CNcccccc6', 's5)))))))cc6', 'ncccc5', '))))))))CC6', 'o5)))))', 'O=CCOC=O)', '-ncccc5', 'NC=O)OCcccccc6))))))))))', 'nc6c%10', 'NCCCCC6))', 'cccBr)', 'CNcnc', 'OCC=O)NCC', 'CCCN5', '))))))))))))))))))))))', 'Cl', '-cccccc6)', 'ccc', 'cccCl)cCl)cc6', 'CC6)))))))cc6', 'CO))', '-ccccs5)))))', 'no5', 'cncc', 'cccc69', 'COccccC=O)', 'cF)cc6', 'COcccccc', '-ccccCl)cc6))))))', 'c=O)[nH]c6=O', 'c6cc%10', 'C)c6', 'n5))))))))', 'C5=O)))))))cc6', 'OC)))))))))', 'C5=O)))))))', 'C=O)NCC)', 'CC6))))))))cc6', 'CCOcccc', 'C)))))', '[nH]ccc95', 'C=O)NCcccccc6', 'cc6c%10', '6OO%11', 'Occc', '-ccccc', 'C=O)O))))))))', 'CCCCCNCC', 'CC%11', 'c6))))))))c6', '))))))))))))))', 'CCOC5', 'Cncn', ')))))))))))))cc6', 'c6n%10', 'cccccc%106))))))))))))', '=S)', 'ccccc6c%10', 'OCCO', 'ccc6[nH]9', 'Nccc', 'CCCCN5', ')CC', 'C)cc5', 'ncn5', '-cnoc', 'COcccc-ccc', '%14', 'cccccF)', 'CCcc', 'cccO)cO)', 'cn96', 'cc96', 'COP=O)O)O', 'NC=O)ccc', 'cncncc6', 'ccn', 'o5))))))))', 'NCC)', 'ccccc', 'CF)F)F)))', 'ccccO)cO)', 'CCO', 'F))', '-nnc', 'F)F)', '8C6', 'NCcccccc6)))))))', 'C=O)N', 'nnn5', '))))))))))cccccc6', 'NCC=O)NCC', 'CccccS=O)=O)', 'N)ncnc6', 'cccccc6))))))', 'cccccccccc%106', 'ncccccc6', '%42', 'CCN)=O))', 'c=O)n', 'cCl)c6))))))', 'c[nH]', 'n95', '%19', 'n6))))))))', 'Cccnc', 'CC=O)cccc', 'cccccCF)F)F))c6', 'CCC=O)N', 'ccC)', 'nc95', 'ccOC))c6', 'CccccF)cc6))))))', 'COC=O)cccc', 'O5', 'CCCC=O)NCC', 'CN5', 'CCN%12', 'nn5-cccccc6', 'ccncc', 'O))))))', 'S', 'ncs', 'CF)F)F))c6))))))', 'CCccccc', '))OCO5', 'cn6)))))))', 'O=CNcccccc6)))))))', 'I', 'ncN)nc6', '-%10', 'Ccc[nH]cccccc96))))))))))', 'NC=O)CScnnc', 'ccccCl)cCl)c6))))))))', 'ccc96', 'CCCC5', 'ccccF)cc6))))))', 'CC5=O', 'S=C', '[Co]', 'CC=O)Nccc', 'cBr)c6', 'CCNCC))', 'NC=O)CC)', 'CNC)cccc', 'CNC)C', 'CCO5', 'oc5C', 'cc6))))))CC6', 'OCF)F', 'COC=O)', 'O)))))', 'C=O)N9', '[n-]', 'ccccCl)cc6', '-ncn', '-cccccc6))))))cc6)))))))', 'CCC5', 'C))))))))', 'nc6', 'NCC=O)', '-cccccCF)F)F))c6))))))', '%35', 'CC=O)NCcccc', 'OC))))', 'NS=O)=O)', 'CC)C)C', '-nn', 'CCC=O)Ncccc', 'CCNCC', 'OCC)C)C', 'CCCNC=O)', '%50', ')))))))))))))))))))', '-cccccCl)c6))))))', 'c5N', 'NCCCCCC6', 'CF)F)F))c6)))))))', 'F)))))))))))', 'cn6', 'CccccO)', 'O)', 'CS=O)=O)', 'NC=O)OCcccccc6', '#N)))', '%23', 'cccccc6)OCO5', 'Ccc[nH]cn5))))))', 'c95', 'O=CCcccc', 'N=CN)', 'OCO5', 'C)C)))', 'cncccc6', 'cc6))))))))cc6', 'cnnnn5', 'nc6))))))))', 'Ncncncc6', 'c%10c%14', 'Ccncc', 'ncccccc%106', 'CC=O)NCC=O)', 'cCl)c6)))))))))', 'N#Cccccc', 'ncccc6c%10', 'CCcccccc6))))))', 'Ccccsc5', 'Cl)c6)))))))', 'S5', '%43', 'CNCcccccc6', 'NC)C))))', 'NC=O)Ncccc', '))))))))))))cc6', 'cc6)))))))))))))', 'ccccO', 'NcncN)', 'C=CC6', 'Occcccc6)', 'CCcccccc6))))))))', 'N%10', '[N+]C)C)', 'C%12=O', 'cccccc6c9', 'NCCCNCC', 'ccccCl)cc6)))))))', 'c5c9', 'cccnc', 'cccncccccc%106', 'C#N))', '))))))cc6', ')))))))))C6', 'c[nH]5', 'C=O)NC)C)))', 'CF)F)CF)F)', 'cccccc6)))))))', 'CCCN', 'nc-cccc', 'NCcccccc6', 'CC=O)NC6=O)))))))', 'NCCNC)', 'cccccO)c6', 'CCCncc', ')))))))))))))))CC6', 'nncc', 'CccccF)cc6)))))))', 'cccccc%106)))))))))))', 'CCCN))))', 'C=O)NC)C', 'CCl)', 'nn', 'nn5', 'CN5C=O)', 'ccC)n', 'nccc=O)[nH]c6=O)))))))', 's5))))))))', 'N#', 'ccccO)cc6))))))', 'cc6)))))))cc6', '85', '-ccnn', 'NCCCCC6', 'cccOC))cOC))', 'Ncccccc6)))))))', 'O=CCS', 'COccccCCNC=O)', '=C5', '))))))))))cc6))))))', 'cF)c6))))))', '-cccc', 'cc6)))))))))', 'CNC=O)ccccc', 'SC)=O)=O', 'CCOC', 'ccOC))c6OC', '5O', 'Ncccnc', '=O))', 'CCNCC))C=O)', 'NCC)=O)))', 'OC', 'CN%10', '=NN5', 'sc', '))))))))', 'CC6))))))cc6', 'CccccNC=O)', ')))))))))c6', 'ccccCl)c', ')', 'cO)', 'cc6))))))cc6', 'NC)C=O)', 'C)cc6))))))))', '[nH]c5', '=CC)', '-ccccF)', 'CC=O)NCC)', 'CN)=O))', 'CC=O)O', 'OCCCNCC', 'Ccnn', 'NC=O)CS', '%18', ')))))))))', 'S=O)=O)cccccc6', 'cO', 'CC)=O', 'CCC=', ')))))))))))CC6', 'C=CC)', 'C9', '-cccccc6))))))cc6))))))', 'C5)))))cc6', '-cccccn6))))))', 'CCCCCCC', 'CCCCOcccc', 'O)))))))))', '%146', 'C=O)OCC', 'NC=O)CC', 'O))))))))', '))))))))))))))))', 'CCCNCcccc', 'CCCNCCC)))', '[nH]cccccc96', 'C=O)O5', 'ccccc[N+]=O)[O-]))c6', 'C)C)))))', 'NC=O)C', 'COC=O)N', '%13%10', 'cncnc', '[nH]nc', 'C=O)O))))', 'ccccF)cc6)))))))', 'OC=O)', '))))))))))))', '=O))))))))))))))))', 'C=O)O)))))', '=N5', 'CC=O)Ncccc', 'N=C', 'CCS', 'C=CCC', 'Ccccn', '[Se]', 'nn6', 'ccncn5', 'NC', 'NCcccccc6))))))', '-cccF)', 'n-cccc', 'ccccC)', 'c6OC', 'SC)=O)=O))', 'CCCCCC6))))))', 'o', 'C6))))))', 'CCCNC)', '#N', 'OC5', 'n6)))))))cc6', 'cnc-cccccc6))))))', 'NCCNCC6', '-cccccc6', 'scc5', 'cnccc', 'o%10', 'NC)C)))', 'CCNC)', 'CO5', 'CC5', 'O=CNCcccc', 'cncccccc6s9', 'NCC)=O', 'ccF)c6', 'COccc', 'Ccccccc6))))))cccccc6', '-cccCl)', 'Cncnc', 'ccCF)F)F))c6', '))))))))))))))CC6', 'CCC)=CCC', '[P+]', 'OCcccccc6))))))))', 'CCcnc', '%10C)', 'ncccN)nc6', 'CCCC4', 'NN=C', 'CccC)', 'CNCCNC=O)', 'F)))))))', '[N-]', 'on5', 'c%139', 'C)=O)=O', 'Ncn', 'CCOC=O)NCC', 'C=O)O)))))))', 'CCcnnc', 'nccnc5', 'cccncc6)))))))', '[P]', 'cccc', 'cccccF)c6', 'cccOC))cOC))cc6', 'CCnc=O)', 'C[N+]C)', '%12', 'C9=O', 'COC=O)cccccc6', 'COcc', 'CF)F)F))cc6)))))))', 'Ccccccc6))))))))))', 'CCC)O)', 'NCcccccc6))))))))', 'OC))', 'cn6)))))))))', '=O)[O-])', 'NCCOCC6', 'NCC)=O))', 'CC=', '%17%14C', 'ccccF)cc6F', '8', 'S=O)=O)O))', 'NCCOCC6)))))))))', 'NC=S)Ncccc', 'cccccc%146', 'CC6)))))))))', 'ccC)cc6', '-5', 'ccO)', 'N))))))))', 'C#Ccccc', 'CNcncc', 'S=O)=O)Ncccc', 'F)', '[2H]', 'CCCC=O)', 'CC)=C', 'c=O)c6', 'Ccnnc', 'NCCO)', 'nC)', 'ncO)', 'c[nH]cccccc6c9', 's5))))))', '[N+]=[N-]', '[O-])', 'ccccnc%106', 'C4))))', '%29', 'CF)F)F))c6))))))))', 's9)))))))))', 'ccc6n9', 'O)))))))', 'Ncc', 'CCOP=O)', '-ccccnc6))))))', 'FCF)F)', 'CN)', 'oc6c%10', 'c9=O', 'NCCCN', '-cnc', 'CCCC7', 'COP=O)', 'NccccCl)', 'n5)))))', 'nC)c5', 'Cl)c6', 'NccccCl)cc6', 'o9', 'cn5', 'c9c%13', 'ccc6c%10', 'CccccC=O)NCC', 'S=O)=O)NCCOCC6)))))))', '-cn', 'CN6', 'cCl)', 'C=', 'CCCC3', 'COcccOC))', '%38', '[o+]', 'CC=CC)C))))', 'cncNcccc', 'P=O)[O-])', '[nH]cc', 'CCO6', '=O)))))))', '-nncc', 'C=O)NCCO', 'CCC)NCC', 'Cl)cc6))))))', 'cccccc6c%10=O', 'ccc6OC', 'C))', 'OCC', 'C)))', 'cccccc6S', 'COcccnc', 'ccc6%10', 'CC)C)))', 'Ncccccc6', 'cns', 'cccccc6C%10', 'CCC3)))))', '[O-]', 'csccc5', 'Ncnccc', 'CCnn', 'cccOC))cOC))cOC))', 'C%11', 'CNCCCCC6)))))))', 'O%10', '))))))))cc6', 'C=O)cccc', 'cC=O)', 'cncccccc6', 'cn5)', 'cccccc', 'cc%106', 'NCCCCC5', 'CO)))', 'cccCF)F)F))ccc6', 'CCCC)', 'nc96', 'NC=O)CNC=O)', 'O=c[nH]', 'N=C5', 'COCCNC=O)', 'cC=O)N', 'COcccOC))cc', 'COcccc-cn', 'NC=O)Ccccc', ')))))))))))c6', 'cccccc6C9=O', 'cncNCC', 'cncccc', 'N#Ccc', 'C=O)NCC=O)', 'c-cccccc6))))))', '=O)))', 'Nccccc', ')C6', ')))))CC6', 'O5)))))', 'cccCl)', 'cccccF)c6))))))))', 'NCCCCCC7', 'O=CCNCC', 'Cc[nH]', 'O=CNC', 'CO', 'C=C5', '%16', 'cccc[N+]=O)[O-]))', 'ccnccc6', 'c6)))))))CC6', '[C-]', 'CCNS=O)=O)cccc', 'C=NCCN5', 'CC))))', 'CNCCOCC6', '))))))))))))))))c6', '-nc', 'CNCC)', 'CCNCCCCC6', 'n5)))))))', '-cnnc', 'Cl)cc6', 'cccC)', 'no5)))))', 'S=O)=O)', 'CCCnc=O)', 'cccNCC', 'c5=O', 'Cl)c', 'CCNCC)', 'C=O)Nccccc', 'cnnC)', 'NccccF)', 'O6', 'ccnC)', 'ccccF)cc6)))))))))', 'CcccccF)', 'nCcccccc6)))))))', 'cnn', 'CNS=O)=O)', 'CCCNS=O)=O)', 'C)C)', 'cO)c6', 'CC4', 'cF)c6))))))))', 'OC))))))))', 'CC9', '[N+]=O)[O-]))', 'ccccO)cc6', '-cc', '=O)[nH]c', '%28', 'COcccOC))c', 'CN))', 'O=CO)', 'C5))))))', 'CCCccc', 'cccccc6))))))))))', '-cccccc6))))))cc6', 'C#N))cc6', '[n+]', 'cncccccc6n9', '[nH]', 'CCC)C)OC=O)NCC', 'cccccc6c%10', 'C=O)NCCCC5', 'cnn5', 'NCCCCC6))))))', 'F))))))))', '=N', 'cccccc6)OCO5)))))))))', 'ccccNCC', 'C=O)ccc', 'CcccC)c', 'CNC)C)))', 'c5)))))', 'CCC)C)O', 'C=O)O))cc6', 'ccO', 'ccccO)cO)c6', 'C)', 'ccCl)', 'C6', 'oc5c9', 'Cccc', 'CNC)', 'ccn6', 'CCO))', ')))))))))))', 'CCN)', 'CCCO', 'C=O)NCcccc', '=C', ')))))))))))))))))', 'cccccO)', 'CCCCO', 'C#N))cc6)))))))', 'Cl))', 'CCNC)C', 'C=N)N', 'NCCCC', 'OCC)C)', 'ccc%146', '96)))))))))', '))))))))))))))))))))))))))))))))', 'ncnnc5', '-nccnc5', 'CF)F)F))))', 'c6)))))))cc6', 'c5C', 'CCC)NC=O)', 'ccccs5', 'cs5', 'ccF)', '[nH]9)))))))))', 'SS', 'ncc6ncn5', 'cccccc96', 'CCCS', 'C5)))))', 'nc=O)', 'OCO5))))))))', 'O=CN', 'CCccc', 'oc%10=O', 'BO)', ')N5', '%45', 'ccCF)F)F))', 'ncNcccc', 'CCOC=O)C=CC)', 'C=CC=O)', 'NC=O)CN', 'O=CCNC=O)', 'Br)', 'cccccCl)c6', 'CC7', 'C=O)Ncccccc6', 'OC))c', 'cccncccCl)ccc%106', '-ccn', 'Cncccc', 'O)cc6', '))))cc6', ')))))))))))))))))cc6', 'Cnc=O)', 'CSS', 'CCC)))', 'CS=O)=O)cccc', 'O=CCC', 'C=O)Nccc', 'cc9', 'ncNCCOCC6))))))', 'CC=CC=O)', 'CCCCC6', 'cn', 'C=O)NCCNCC', 'cnc', 'CC5))))))', 'ccccBr)', 'C#', 'CNC=O)', '-ccnnC)c5)))))', 'N)=O)))', 'cccO)ccc6', 'Occcccc6', 'S=O)=O)O)))', 'cno', '-ccs', 'cn6))))))', 'CS', 'ncn6', '%17', 'CCC)CC=O)', 'ccc-cccccc6))))))', '))))))))))))))cc6', 'NC=O)cccc', 'CCOC=O)cc', 'CCNS=O)=O)', 'ncnc69', '-cccnc', 'o5))))))', 'cOC))', 'cccccc%106))))))))))', 'C=O)N5', 'C=N)N))', 'csc', 'COccccS=O)=O)', 'Ccccc-cccccc6', '%49', 'CCCC))))', 'ncccccc6c%10=O', 'cCl)c6)))))))', ')))))', 'CCNC', 'CCOccc', '-ccn[nH]c5', 'C)))))))))))', 's', 'n5))))))', 'c-cccc', 'O))))', 'n5', 'CF)F)F)))))))))', 'ccccccc6', 'C=O)NCCCCCC6', 'ccccF)cc6))))))))', 'cnc6', 'O=[N+][O-])', 'CccccCl)cc6', 'NC=O)CNCC', 'ncC)', '))', 'nnc5S', 'ncnc6', 'CCCCC7', 'CF)F)F))cc6))))))', 'B', 'CC', 'nc69', 'CC=O)N', '))))))))))))c6', 'C%10', 'CNCCCCC6', 'CF)F)F))cc6', 'C=O)NCCOCC6)))))))', '[nH]c6=O', 'CccccO', '-cccn', 'n-cccccc6))))))', 'CC6))))))', 'cC)', 'n5)))))cc6', '))))C6', 'NcccCF)F)F))', 'CCCN6', '[N+]=O)[O-]))cc6))))))', 'C)C))))', ')))))))))))))))))))))', 'Fcccc', '-cccsc5)))))', ')))', '[nH]cccccc69', 'ccc6n%10', 'cccCl)ccc6O', '-cccccc6))))))', 'cccccccc6', 'Ccccccc6)))))))))', 'ncc5', 'NC=O)OCC)C)C)))))', '=O))))))', 'CCC=O)NCC', 'CNCccccc', '[nH]n5)))))', '=C6', 'NCCNcccccc6', 'CC#', 'N%11', 'n5-cccccc6', 'C%14', '-cccccc6)OCO5))))))))', 'OCC)))', '-cccccc6F)))))))', 'cc=O)', '-cnccc', 'C=CCNC=O)', '-cccncc6', 'CF)F)F', 'CC)C)C)))))', 'C)C6', 'ccccO)cc6))))))))', 'ncNCC', 'n6))))))', 'NC=O)Nccccc', '[SH]', ')))))cc6', 'c6c%10', 'COCC)=O))))', 'O=CNccccc', 'cccO)c6', 'C=O)NS=O)=O)', 'SC)=O)=O)))', '[N+]=O)[O-]))cc6', 'ccc=O)', 'OCF)F)))', 'cC=O)O))cn', 'cC)cc6', '-ccccCF)F)F))cc6))))))', 'cccccc6n%10', 'ccc5', 'NCCCCCC', 'c96', ')))))))))cc6OC', 'CC=O)O))))', '%36', 'cO)cc6', 'c6c%10=O', 'NC=O)ccccc', 'C5))))))))', 'cnccs5', 'Ccn', '-cnn', 'c=O)n6C', 'CC6', 'OC))cc6', 'Ccccccc6)))))))', 'CNC=O)cccccc6', 'CNC=O)ccc', 'NS=O)=O)cccccc6', 'OC=O)cccccc6))))))))', '%14%17C', 'CC)C)))))', '[nH]cc5', 'CNC=O)cccc', 'ccc6)', 'OCCO)', 'NC=S)NCC', '))))))))))))))))))))', 'CCCN5C=O)', 'N', 'COccccC=O)N', 'Cl)))))))', 'N)N', 'NCCO)))', 'O', 'c5', 'cn5))))))', 'ncccN)nc6=O)))))))', '[As]', 'cnccn5', 'OCCC', 'c6)))))))', 'COCC)=O', 'OCccccc', ')))))))))n6', 'cccccc69', 'C)cc6', 'cccccCl)c6))))))))', 'ncccc', 'C5=O))))))', '%33', 'CccccF)', 'ccC)nn', '9O', 'CF)F)F))c6', '))))))))))cc6', '))))))))))))))))))))))))))', 'S=O)=O)NCCO', 'ncnccN)ncnc6', 'C))))))', 'CCCCNCC', 'cccccc6)', 'COCC', 'c6))))))c6', 'CO3', 'cncc-cccc', '))))))))))))))))))', '[N+]=O)[O-]))cc6))))))))', 'CCOC=O)', ')))))))))))))CC6', '%15', 'CccccCl)', 'CC)', 'CCOCCO', 'cncN)', 'cc', 'nc%106', 'C=CCNCC', '[N+]=O)[O-]', 'n%10', 'cccccc6)))))))))', '5', ')))))))cc6', 'c=O)[nH]', '[nH]5)))))', 'c6))))))))cc6', 'cccccc6OC', 'F)))', 'cF)c6', 'CCOcccccc6', 's5', 'CC3))))', 'cn5)))))', '[se]', 'C5=O', 'CCNCC=O)', 'C=O)cccccc6)))))))', 'CC3', '[N+]=O)[O-]))cc6)))))))', '=S', 'Br)cc6))))))))', 'C))))))))))))', 'F)F', 'CCC5)))))))', 'Ncccn', 'OCcccccc6', 'C=O))', 'cN)', 'c%14', '[s+]', '=O))))))))', 'cccccc6', '%17C', 'NCCCC5', '6))', '%37', 'NCccccc', 'COCCO', 'CCCC)C)', 'ccF)cF)', '))))))))))))))c6', '%34', '%32', 'CNCCNcccc', '))))))))))C6', 'c=O)n6', '6O', 'NS=O)=O)ccccc', 'O%11', 'NCCNCC6))))))', 'NCCO', 'O=CNCC', 'OCF)F)F)))cc6', 'CCCNcccc', '%106', 'CCnnc', 'CNCCO', 'O)c', 'CccccCl)cc6))))))', 'C3))))', 'nC)c6=O', 'ccCl)cc6', '6))))))', 'NCN)', 'CCCCCCCC=O)', 'CCNCcccc', 'oc%106', '[Na]', 'n6))))))cc6', 'c9C', '-6', '))))))))))cc6OC', ')))))C6', 'CCOCCOCCOCCO', '-cccncc6))))))', 'cc[nH]cccccc96', '[nH]c5c9', 'CCC4', 'CCC=O)', 'C#N))c', 'cCl)cc6', 'Ccccccc6))))))))', ')))))))', 'cC=O)O))cn6', 'NCCNCC', 'nn95', 'cccCl)ccc%106', '))))))))))n6', 'C3', 'C))))', '9)))))))))', ')))))))))))))', '-cccccc6Cl)))))))', 'Ncncccccc6', 'ccccO)', 'O=Ccccccc6', 'nccC)c=O)[nH]c6=O)))))))', 'n9', 'CCC)C)))', 'CC)C)C))cc6', 'C)c5', 'CC6)))))', 'S=O)=O)NCC', 'cC=O)O))', 'ccccc6', 'SC5', '))))))CC6', 'ccoc', 'CCNCcccccc6)))))))', '))))))))))))CC6', 'CC=O)NC', 'oc', '))))))))))c6', 'CF)F))', 'CNCcccc', 'CC6)))))))', 'C%11=O', '=[N+]', 'P=O)', 'C=CC', 'C))))))))))', '[nH]ncc5', 'c6)))))))))', 'ncN)', 'cccC=O)NCC', 'cccsc5', 'CS=O)=O)NCC', 'CCCCNC=O)', '))))))))))CC6', 'cccc%106', '-ccnnC)c5', 'COcccccc6OC))))', 'cBr)', 'cnc95', 'C)))))))))', '95', ')))))))CC6', 'c=O)', ')))))c6', 'CCCCC5', 'cncn', 'Cccccc', 'CCCN=CN)N))))))', 's8', 'CC=O)O))', 'Scccccc6', 'cccF)ccc6', '-ccccnc6', 'CF)F)', 'ccOC))', 'O=C', 'ccN)ncnc6', 'N=', 'cccCF)F)F))', 'CCC)', 'ccnccn6', '%27', 'n[nH]c5', 'CCCCCCCC6)C8)))C6', 'CCN)=O)))', 'C)c', 'OCCO5', '6)))))))', 'COccccc', 'F)cccc6', 'CCCCN6', 'c=O)c%10', 'CNCC=O)', 'Ncncncc6ncn5', 'CCS=O)=O)', 'CCC', '-cccsc5', 'cccccc6))))))cccccc6', 'CNcccc', 'CCO)))', 'P=O)O)', 'C=O)NCC=O)NCC', '[S+]', 'CCCCC', 'C)))))))', 'COccccF)cc6', 'C6)))))))', 'OCC=O)N', 'ccccF)', 'Br', ')))))))))))))))))))))))))', 'nc5c9', 'C#N', 'COcccc-', 'CF)F)F))))))))', '%25', 'N9', 'c', 'cn5C))))))))', 'Ccccco5', 'O=ccc', 'c[nH]c', 'N=CN)cccc', '=O)[nH]cN)nc6', 'O=CO)cccc', 'ncncc', 'OCCCC', 'CccccO)cc6', 'CCC3', 'CO)C6', 'NcccccCl)c6', 'c=O)[nH]6', 'occc', ')))))))))cc6', 'cncncN)', '%14C', 'P', 'Cl))))))))))', 'nnc5', 'Ccncccccc6', 'C6C)C', 'C=O)NCCCCC6', 'CccccS=O)=O)NCC', '7', 'cccc-cccc', '%46', '6))))))))', 'CCCCCCC6', '[PH]', 'ccccc6n%10', '-ccccCl)', 'CccccO)cc6)))))))', 'n5)))))c6', 'F)F)F', 'cc6))))))', 'cccc[N+]=O)[O-]))cc6', 'NCcccc', 'CCCO5', 'cccccc6F', 'Ncnc', '))))))))))ccc6', 'SC))', '))))', 'Occcccc6)))))))', 'OC))cOC))', 'CCCNC', '96', 'C8', 'cccoc5', 'ccc5c9', 'nc%10', 'cccccCl)c6))))))', 'CCCCNC', 'NCCNC', 'cF)', ')))))))C6', 'CC6))))))))', 'CC)C))))', 'cCl)c6', 'sc5c9', 'cncC)', '-ccncc', 'Ccccnc', 'cc6))))))))c6', 'Clccccc', 'CCCCcccc', 'NCCNcccc', 'cccncc6', 'c%10=O', 'O=CCO', 'CC)C))))))))', 'COcccO)']
    
    
    print(deepsmiles_ids_tuple[0])
            
        

    return result_deepsmiles_tokens
    
def corrupt_tokens_with_custom_mask(mass_token_ids, mask_token_id, mask_ratio=.2):
    num_to_mask = int(len(mass_token_ids) * mask_ratio)
    mask_indices = random.sample(range(1, len(mass_token_ids)), num_to_mask)
    for idx in mask_indices: 
        mass_token_ids[idx] = mask_token_id
        
    return mass_token_ids
def tokenize_spectrum(exact_mass, spectrum): 
    mz_tokenizer_map = {np.round(mz, 1): idx + 3 for idx, mz in enumerate(np.arange(0, 1000, .1))} # + 2 because model's last 
    # inherent token is EOS tk which is = 2, 
    mz_tokenizer_map[1000] = 10003 # should be +4 not +3 after this line
    mass_tokenized = [] 
    mass_tokenized.append(mz_tokenizer_map[exact_mass])
    # make sure its rounded up first 
    
    for i in range(len(spectrum)): 
        #expect float input
        if len(mass_tokenized) == 256: 
            break
        mass_tokenized.append(mz_tokenizer_map[spectrum[i][0]])
        
    
    # mass_tokenized = corrupt_tokens_with_custom_mask(mass_tokenized, 10003, 0.9)

    if (256 - len(mass_tokenized)) >= 0:
        for i in range(256 - len(mass_tokenized)):
            mass_tokenized.append(1)

    return mass_tokenized
    
def str_to_float(str_spectrum): 
    # print(f"len str_spec: {len(str_spectrum)}")
    float_spectrum = [] 
    
    left, right = 1, 0
    # print(len(str_spectrum))
    while(right < len(str_spectrum) - 1): 

        # print(str_spectrum[right])
        if str_spectrum[right] == "]": 
            while (str_spectrum[left] != "["): 
                left += 1 
            str_tuple = str_spectrum[left + 1: right] 
            x, y = str_tuple.split(',')
            float_spectrum.append([np.round(float(x), 1), float(y)])                                   
            # print(x, y)
            left = right 
            
            right += 1
        right += 1

    # print(len(float_spectrum))

    return float_spectrum 


# In[ ]:


from torch.utils.data import DataLoader, Dataset
import torch

class CustomDataset(Dataset):
    def __init__(self, exact_mass, encoder_spectrum_preTK, decoder_smiles_preTK, encoder_tokenizer, decoder_tokenizer): 
        self.exact_mass = exact_mass
        self.encoder_spectrum_preTK= encoder_spectrum_preTK
        self.decoder_smiles_preTK = decoder_smiles_preTK
        self.encoder_tokenizer = encoder_tokenizer 
        self.decoder_tokenizer = decoder_tokenizer

    def __len__(self): 
        return len(self.encoder_spectrum_preTK)

    def __getitem__(self, idx):


        float_exact_mass = np.round(self.exact_mass[idx], 1)
        float_spectrum = str_to_float(self.encoder_spectrum_preTK[idx])
        # if idx == 0: 
            # print(len(self.encoder_spectrum_preTK[idx]))
            # print(len(float_spectrum))
        _, float_intensity_tuple = zip(*float_spectrum)
        float_intensity_list = list(float_intensity_tuple)
        # print(float_spectrum)
        # print(float_intensity)
        float_intensity_list.insert(0, 2.0) # for the exact mass, doens't have theoerical intensity so must use 2.0
        float_intensity_list_padded = float_intensity_list 
        if len(float_intensity_list) == 257: 
            float_intensity_list.pop()
        for i in range(256 - len(float_intensity_list)): 
            if len(float_intensity_list) == 256: 
                break
            float_intensity_list_padded.append(1)
        encoder_mass_tokenized = self.encoder_tokenizer(float_exact_mass, float_spectrum) # haven't turned data into 
        encoder_mass_tokenized_tensor = torch.tensor(encoder_mass_tokenized)
        # float yet
        encoder_attention_mask = (encoder_mass_tokenized_tensor != 1).long()
        # encoder_attention_mask_with_random = random_mask_exact_mass(encoder_attention_mask)
        encoder_attention_mask_tensor = torch.tensor(encoder_attention_mask)
        
        decoder_smiles_tokenized, decoder_label = self.decoder_tokenizer(self.decoder_smiles_preTK[idx])
        decoder_label_tensor = torch.tensor(decoder_label)
        decoder_smiles_tokenized_tensor = torch.tensor(decoder_smiles_tokenized)
        
        
        decoder_attention_mask = (decoder_smiles_tokenized_tensor != 1).long()
        
        decoder_attention_mask_tensor = torch.tensor(decoder_attention_mask)

        # print(f"len enc tens:{len(encoder_mass_tokenized)}")
        # print(f"len att:{len(encoder_attention_mask)}")
        # print(f"len dec tens:{len(decoder_smiles_tokenized)}")
        # print(f"len att:{len(decoder_attention_mask)}")
        
        # print(f"len intens:{len(float_intensity_list)}")

        return { 'encoder_input_id': encoder_mass_tokenized_tensor, 'encoder_attention_mask': encoder_attention_mask_tensor, 
                'decoder_input_id': decoder_smiles_tokenized_tensor, 'decoder_attention_mask': decoder_attention_mask_tensor, 'decoder_label': decoder_label_tensor, 
                'intensity': torch.tensor(float_intensity_list) } 


# In[ ]:


data = pd.read_json("./clean_msms_normalized_sorted_exactMass.json")


spectrum = data['sorted_peaks'].tolist()
deepsmiles = data['deepsmiles'].tolist()
exact_mass = data['ExactMass'].tolist()

dataset = CustomDataset(exact_mass, spectrum, deepsmiles, tokenize_spectrum, tokenize_deepsmiles)

#dataloader = DataLoader(dataset, batch_size=1024, shuffle=True, pin_memory=True, num_workers=192) 

import torch.nn as nn
from transformers import BartModel, BartTokenizer, AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import BartConfig
from torch.cuda.amp import autocast, GradScaler

class CustomBARTModel(nn.Module):
    def __init__(self, bart_model):
        super(CustomBARTModel, self).__init__()
        self.bart = bart_model
        
        # Separate embedding layers for encoder and decoder
        self.encoder_embedding = nn.Embedding(10000 + 4, 768)
        self.decoder_embedding = nn.Embedding(269 + 3, 768)
        # Replace the default shared embeddings in BART
        self.bart.encoder.embed_tokens = self.encoder_embedding
        self.bart.decoder.embed_tokens = self.decoder_embedding
        
        # Fully connected layers for encoder
        # self.fc1 = nn.Linear(769, 768)
        # self.fc2 = nn.Linear(768, 768)
        self.fc1 = nn.Linear(769, 768)
        self.fc2 = nn.Linear(768, 768)
        self.fc_logits = torch.nn.Linear(self.bart.config.d_model,269 + 3 )



    def forward(self, encoder_input_ids, intensity, decoder_input_ids, encoder_attention_mask=None, decoder_attention_mask=None):
        # Encoder: Get embeddings and concatenate extra tensor
        
        encoder_embedded = self.encoder_embedding(encoder_input_ids)
        
        intensity = intensity.unsqueeze(-1)
        # print(f"encoder_embedded shape: {encoder_embedded.shape}")
        # print(f"intesniry shape: {intensity.shape}")
        combined_encoder_embedded = torch.cat((encoder_embedded, intensity), dim=-1)
        combined_encoder_embedded = self.fc1(combined_encoder_embedded)
        combined_encoder_embedded = torch.relu(combined_encoder_embedded)
        combined_encoder_embedded = self.fc2(combined_encoder_embedded)

        
        
        # Forward pass through BART with modified embeddings
 

        encoder_outputs = self.bart.encoder(
            inputs_embeds=combined_encoder_embedded,
            attention_mask=encoder_attention_mask
        )
    
     
        
        # Decoder: Get embeddings
        decoder_embedded = self.decoder_embedding(decoder_input_ids)
        
        # Pass through the decoder
        decoder_outputs = self.bart.decoder(
            inputs_embeds=decoder_embedded,
            encoder_hidden_states=encoder_outputs[0],  # Pass encoder outputs to decoder
            attention_mask=decoder_attention_mask,
            encoder_attention_mask=encoder_attention_mask, 
            use_cache=False
        )
        logits = self.fc_logits(decoder_outputs.last_hidden_state)
        return logits


import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import BartModel, BartConfig, AdamW
import os
import random  # Ensure you import random if not already imported
from torch.cuda.amp import autocast, GradScaler  # Import AMP modules

# Setup function to initialize distributed process group
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'  # For single-node training
    os.environ['MASTER_PORT'] = '29500'      # Use any open port (e.g., 29500)
    
    dist.init_process_group(
        backend='nccl',  # 'nccl' for GPUs
        init_method='env://',  # URL to initialize the process group
        world_size=world_size,  # Total number of processes (i.e., number of GPUs)
        rank=rank  # Rank of the current process
    )

# Cleanup function to destroy the process group
def cleanup():
    dist.destroy_process_group()

# Training loop function
def train(rank, world_size, config, dataset):
    # Set up the distributed process group
    setup(rank, world_size)

    # Set up the device for the current process
    device = torch.device(f'cuda:{rank}')

    # Initialize the model and move it to the correct device
    bart_model = BartModel(config)
    custom_bart_model = CustomBARTModel(bart_model).to(device)
    # Wrap the model in DistributedDataParallel
    ddp_model = DDP(custom_bart_model, device_ids=[rank])

    # Define optimizer
    optimizer = AdamW(ddp_model.parameters(), lr=1e-4)

    # Initialize GradScaler for AMP
    scaler = GradScaler()

    # Use DistributedSampler to ensure each GPU gets its own part of the dataset
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)

    # Create DataLoader using the distributed sampler
    dataloader = DataLoader(
        dataset, 
        batch_size=128,  # Adjust batch size based on GPU memory
        shuffle=False,   # Don't shuffle, DistributedSampler takes care of it
        num_workers=12,  # Adjust based on your system; reduce if memory issues occur
        pin_memory=True,
        sampler=sampler, 
        drop_last=True,
        persistent_workers=True, 
        prefetch_factor=4, 
        
    )

    # Training loop
    for epoch in range(50):
        ddp_model.train()
        if epoch < 10: 
            teacher_forcing_ratio = 1
        else:
            teacher_forcing_ratio = 1 - (0.025 * (epoch - 9)) 

        sampler.set_epoch(epoch)  # Ensures each GPU gets different data each epoch

        for batch in dataloader:
            # Check for NaNs or Infs in the batch
            for name, tensor in batch.items(): 
                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                    print(f"Invalid value in {name}")
                    print(tensor)
                    nan_mask = torch.isnan(tensor) 
                    inf_mask = torch.isinf(tensor)
                    print(nan_mask.nonzero(as_tuple=True))
                    print(inf_mask.nonzero(as_tuple=True))
                    print(tensor[nan_mask])
                    print(tensor[inf_mask])

            optimizer.zero_grad()

            # Move tensors to device
            encoder_input_ids = batch['encoder_input_id'].to(device, non_blocking=True)
            encoder_attention_mask = batch['encoder_attention_mask'].to(device, non_blocking=True)
            decoder_input_ids = batch['decoder_input_id'].to(device, non_blocking=True)
            decoder_attention_mask = batch['decoder_attention_mask'].to(device, non_blocking=True)
            decoder_label = batch['decoder_label'].to(device, non_blocking=True)
            intensity = batch['intensity'].to(device, non_blocking=True)

            use_teacher_forcing = random.random() < teacher_forcing_ratio

            # Enable autocast for mixed precision
            with autocast():
                if not use_teacher_forcing:
                    # Initialize generated_ids with the decoder start token
                    generated_ids = torch.full(
                        (decoder_input_ids.size(0), 1), 
                        custom_bart_model.bart.config.decoder_start_token_id, 
                        dtype=torch.long
                    ).to(device, non_blocking=True)
                    
                    # Generation loop
                    for t in range(decoder_input_ids.size(1)):
                        outputs = custom_bart_model(
                            encoder_input_ids=encoder_input_ids,
                            intensity=intensity,
                            decoder_input_ids=generated_ids,
                            encoder_attention_mask=encoder_attention_mask,
                            decoder_attention_mask=None, 
                        )

                        # Get the next token logits and select the argmax token
                        next_token_logits = outputs[:, -1, :]
                        next_token = next_token_logits.argmax(dim=1, keepdim=True).detach()
                        generated_ids = torch.cat([generated_ids, next_token], dim=1)

                        # Break if all sequences have generated EOS token
                        if (next_token == custom_bart_model.bart.config.eos_token_id).all():
                            break

                    logits = outputs

                    # Adjust logits and labels to have matching sequence lengths
                    generated_length = generated_ids.size(1)
                    target_length = decoder_input_ids.size(1)

                    if generated_length < target_length:
                        padding_length = target_length - generated_length
                        padding_tensor = torch.full(
                            (logits.size(0), padding_length, logits.size(-1)), 
                            fill_value=-65504.0, 
                            dtype=logits.dtype, 
                            device=logits.device
                        )
                        logits = torch.cat([logits, padding_tensor], dim=1)
                    elif generated_length > target_length:
                        logits = logits[:, :target_length, :]

                    labels = decoder_label  # Labels remain the same

                else:
                    # Teacher-forcing mode
                    outputs = custom_bart_model(
                        encoder_input_ids=encoder_input_ids,
                        intensity=intensity,
                        decoder_input_ids=decoder_input_ids,
                        encoder_attention_mask=encoder_attention_mask,
                        decoder_attention_mask=decoder_attention_mask, 
                    )
                    logits = outputs
                    labels = decoder_label  # Labels remain the same

                # Ensure logits and labels have matching sequence lengths
                seq_length_logits = logits.size(1)
                seq_length_labels = labels.size(1)

                if seq_length_logits < seq_length_labels:
                    padding_length = seq_length_labels - seq_length_logits
                    min_value = -65504.0  # Use a smaller negative value
                    padding_tensor = torch.full(
                        (logits.size(0), padding_length, logits.size(-1)),
                        fill_value=min_value,
                        dtype=logits.dtype,
                        device=logits.device
                    )
                    logits = torch.cat([logits, padding_tensor], dim=1)
                elif seq_length_logits > seq_length_labels:
                    # Truncate logits to match the length of labels
                    logits = logits[:, :seq_length_labels, :]

                # Flatten logits and labels
                logits = logits.view(-1, logits.size(-1))
                labels = labels.view(-1)

                # Compute loss
                pad_token_id = custom_bart_model.bart.config.pad_token_id
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id).to(device, non_blocking=True)
                loss = loss_fct(logits, labels)

            # Check for invalid loss
            if not torch.isfinite(loss):
                print(f"Skipping step due to invalid loss: {loss}")
                continue

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if rank == 0:  # Only log from rank 0 process
                print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    # Cleanup the process group
    cleanup()

# Main function to set up distributed training
def main():
    world_size = torch.cuda.device_count()  # Number of available GPUs
    print(f"Number of GPUs: {world_size}")

    config = BartConfig(
        d_model=768,
        encoder_layers=6,
        decoder_layers=3,
        encoder_attention_heads=12,
        decoder_attention_heads=6,
        bos_token_id=0,
        pad_token_id=1,
        eos_token_id=2,
        mask_token_id=10003,
        decoder_start_token_id=0,
        max_position_embeddings=256,
        encoder_ffn_dim=3072,
        decoder_ffn_dim=3072
    )

   
    # Use mp.spawn to start multiple processes, each corresponding to a GPU
    mp.spawn(train, args=(world_size, config, dataset), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()



