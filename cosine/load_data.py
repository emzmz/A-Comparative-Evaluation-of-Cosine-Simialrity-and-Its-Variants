import os
import pandas as pd
import config

def load_data():
    base_path = config.RAW_DATA_PATH
    datasets = {}

    # WordSim535
    ws_path = os.path.join(base_path, "wordsim353crowd.csv")
    ws = pd.read_csv(ws_path)
    ws.columns = ['word1', 'word2', 'human_score']
    datasets["WordSim353"] = ws

    # SimLex999
    simlex_path = os.path.join(base_path, "SimLex-999.txt")
    simlex = pd.read_csv(simlex_path, sep='\t')
    simlex.columns = ['word1', 'word2', 'human_score']
    datasets["SimLex999"] = simlex

    # MC30
    mc30_path = os.path.join(base_path, "mc30.csv")
    mc30 = pd.read_csv(mc30_path, sep=';', header=None)
    mc30.columns = ['word1', 'word2', 'human_score']
    datasets["MC30"] = mc30

    # MEN
    men_path = os.path.join(base_path, "MEN_dataset_natural_form_full")
    men = pd.read_csv(men_path, sep=' ', header=None, usecols=[0, 1, 2])
    men.columns = ['word1', 'word2', 'human_score']
    datasets["MEN"] = men

    # MTurk
    mturk_path = os.path.join(base_path, "mturk.part.json")
    mturk = pd.read_json(mturk_path)
    mturk = mturk.iloc[:, :3] 
    mturk.columns = ['word1', 'word2', 'human_score']
    datasets["MTurk"] = mturk

    return datasets
