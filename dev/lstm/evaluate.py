import pickle
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

def main(validation_data_path, cm_out_path):

    genres_to_int = {'metal': 0, 'rap': 1, 'rock': 2, 'dance': 3, 'alternative': 4}

    validation_data = pickle.load(open(validation_data_path, 'rb'))
    logits, targets = validation_data['predictions'], validation_data['targets']

    predictions = logits.argmax(axis=1)

    cls_rpt = classification_report(targets, predictions, target_names=genres_to_int.keys())
    cm = confusion_matrix(targets, predictions, normalize='true')

    df_cm = pd.DataFrame(cm, index = genres_to_int.keys(), columns = genres_to_int.keys())
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)

    plt.savefig(cm_out_path)
    print(cls_rpt)



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('validation_data_path')
    parser.add_argument('cm_out_path')

    main(**vars(parser.parse_args()))