import torch

import src.constants as constants
from src.data_loader import load_data

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

midi_data, train_dataset, test_dataset, valid_dataset = load_data()

model = torch.jit.load("result_model/satb_forward_network.pt")
print(model)

input_names = [ "input" ]
output_names = [ "forward_alto", "forward_tenor", "forward_bass" ]

(x_soprano_sample, y_alto, y_tenor, y_bass) = train_dataset[0:constants.BATCH_SIZE]
torch.onnx.export(model, x_soprano_sample.to(device), "result_model/satb_forward_network.onnx", verbose=True, input_names=input_names, output_names=output_names)