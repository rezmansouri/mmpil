import os
import sys
import h5py
import torch
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import roc_curve
from aeon.classification.interval_based import TimeSeriesForestClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-9


def pad_and_resize(image, size=(128, 128)):
    '''empty'''
    h, w = image.shape
    v_pad = max(0, w-h) // 2
    h_pad = max(0, h-w) // 2
    padded = np.pad(image, ((v_pad, v_pad), (h_pad, h_pad)))
    img = Image.fromarray(padded * 255)
    img = img.resize(size, Image.LANCZOS)
    img = np.array(img)
    img[img != 0] = 1
    return img


def get_raw_data(paths):
    '''empty'''
    print(f'reading from {paths}')
    X_properties, X_pil, Y = [], [], []
    for path in paths:
        for h5_name in tqdm(os.listdir(path), leave=False):
            with h5py.File(os.path.join(path, h5_name), 'r') as h5:
                properties = h5.get('properties')[:]
                X_properties.append([
                    [float(item[1].decode()) if item[1].decode() != '' else np.NaN for item in p] for p in properties
                ])
                X_pil.append(pad_and_resize(h5.get('pil')[:]))
                class_ = h5.get('class')[()]
                y = 1 if class_ in [b'X', b'M'] else 0
                Y.append(y)
    return np.array(X_properties, dtype=np.float32), np.array(X_pil, dtype=np.float32), np.array(Y, dtype=np.float32)


class PILModel(nn.Module):
    '''empty'''

    def __init__(self, input_dim=128, in_channels=1, out_channels=4, latent_dim=128):
        super(PILModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding='same')
        self.conv2 = nn.Conv2d(
            out_channels, out_channels * 2, 3, padding='same')
        self.flatten_dim = (input_dim // 4) ** 2 * out_channels * 2
        self.fc1 = nn.Linear(self.flatten_dim, self.flatten_dim//4)
        self.fc2 = nn.Linear(self.flatten_dim//4, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 1)
        self.max = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        '''empty'''
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max(x)
        x = x.view(-1, self.flatten_dim)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


def tss(yhat, y):
    '''empty'''
    yhat = np.array(yhat)
    y = np.array(y)
    tp = ((y == 1) & (yhat == 1)).sum()
    fn = ((y == 1) & (yhat == 0)).sum()
    fp = ((y == 0) & (yhat == 1)).sum()
    tn = ((y == 0) & (yhat == 0)).sum()
    score = (tp / (tp + fn + EPS)) - (fp / (fp + tn + EPS))
    return score


def hss(yhat, y):
    '''empty'''
    yhat = np.array(yhat)
    y = np.array(y)
    tp = ((y == 1) & (yhat == 1)).sum()
    fn = ((y == 1) & (yhat == 0)).sum()
    fp = ((y == 0) & (yhat == 1)).sum()
    tn = ((y == 0) & (yhat == 0)).sum()
    numerator = 2 * (tp * tn - fp * fn)
    denominator = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn) + EPS
    score = numerator / denominator
    return score


def image_predict(x, y, loss_fn, model):
    '''empty'''
    with torch.no_grad():
        out = model(x)
        loss = loss_fn(out, y).item()
        return out, loss


def interpolate_properties(X):
    '''empty'''
    for ix, x in enumerate(X):
        if not np.isnan(x).any():
            continue
        needle = 0
        while needle < len(x):
            t = x[needle]
            if np.isnan(t).any():
                prev_t = x[needle-1]
                next_t = None
                n_missing = 1
                for temp_next_t in x[needle+1:]:
                    if not np.isnan(temp_next_t).any():
                        next_t = temp_next_t
                        break
                    else:
                        n_missing += 1
                diff = next_t - prev_t
                step = diff / (n_missing + 1)
                for i in range(needle, needle+n_missing):
                    X[ix, i, :] = X[ix, i-1, :] + step
                needle += n_missing + 1
                continue
            needle += 1
    return X


def main():
    '''empty'''
    if len(sys.argv) != 7:
        raise ValueError(
            'usage: python m1_m2.py {partitions root} {training partitions} {validation partitions} {testing partitions} {pil model path} {output path}\nexample: python main.py /data 135 2 4 ./pil_model.pt ./out')
    root, training_partitions, validation_partitions, testing_partitions, pil_model_path, output_path = sys.argv[
        1:]
    train_paths = [os.path.join(
        root, f'partition{path}') for path in training_partitions]
    val_paths = [os.path.join(
        root, f'partition{path}') for path in validation_partitions]
    test_paths = [os.path.join(
        root, f'partition{path}') for path in testing_partitions]

    x_properties_train, _, y_train = get_raw_data(train_paths)
    x_properties_val, x_pil_val, y_val = get_raw_data(val_paths)
    x_properties_test, x_pil_test, y_test = get_raw_data(test_paths)

    x_properties_train = interpolate_properties(x_properties_train)
    x_properties_val = interpolate_properties(x_properties_val)
    x_properties_test = interpolate_properties(x_properties_test)

    x_properties_train = np.transpose(x_properties_train, [0, 2, 1])
    x_properties_val = np.transpose(x_properties_val, [0, 2, 1])
    x_properties_test = np.transpose(x_properties_test, [0, 2, 1])

    properties_model = TimeSeriesForestClassifier()

    print('training tsf')
    nf_ix = np.where(y_train == 0)[0]
    f_ix = np.where(y_train == 1)[0]

    undersampled_nf_train_ix = np.random.choice(nf_ix, len(f_ix), False)
    x_properties_train_balanced = x_properties_train[np.append(
        f_ix, undersampled_nf_train_ix)]
    y_train_balanced = y_train[np.append(f_ix, undersampled_nf_train_ix)]

    properties_model.fit(x_properties_train_balanced, y_train_balanced)

    pil_model = PILModel().to(DEVICE)
    pil_model_state = torch.load(pil_model_path, map_location=DEVICE)
    pil_model.load_state_dict(pil_model_state)
    pil_model.eval()

    x_pil_val = np.expand_dims(x_pil_val, axis=1)
    x_pil_val = torch.tensor(x_pil_val, dtype=torch.float32).to(DEVICE)

    x_pil_test = np.expand_dims(x_pil_test, axis=1)
    x_pil_test = torch.tensor(x_pil_test, dtype=torch.float32).to(DEVICE)

    properties_yhat_val = properties_model.predict_proba(x_properties_val)[
        :, 1]
    with torch.no_grad():
        pil_yhat_val = pil_model(x_pil_val).cpu().detach().numpy().squeeze()
    fused_yhat_val = np.mean(
        np.stack((properties_yhat_val, pil_yhat_val)), axis=0)

    fpr, tpr, thresholds = roc_curve(y_val, fused_yhat_val)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    print('optimal threshold:', optimal_threshold)

    stats = {}

    pred_val = fused_yhat_val
    pred_val[pred_val < optimal_threshold] = 0
    pred_val[pred_val >= optimal_threshold] = 1

    stats['val_hss'] = hss(pred_val, y_val)
    stats['val_tss'] = tss(pred_val, y_val)

    properties_yhat_test = properties_model.predict_proba(x_properties_test)[
        :, 1]
    with torch.no_grad():
        pil_yhat_test = pil_model(x_pil_test).cpu().detach().numpy().squeeze()
    fused_yhat_test = np.mean(
        np.stack((properties_yhat_test, pil_yhat_test)), axis=0)

    pred_test = fused_yhat_test
    pred_test[pred_test < optimal_threshold] = 0
    pred_test[pred_test >= optimal_threshold] = 1

    stats['test_hss'] = hss(pred_test, y_test)
    stats['test_tss'] = tss(pred_test, y_test)

    df = pd.DataFrame(stats, index=[0])
    print(df.head())
    df.to_csv(
        os.path.join(output_path, 'm1m2.csv'), index=False)


if __name__ == '__main__':
    main()
