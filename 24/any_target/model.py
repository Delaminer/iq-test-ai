import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class TensorModel(nn.Module):
    def __init__(self):
        super(TensorModel, self).__init__()
    def split_data(self, solutions):
        separate_operation_indices = True
        label_is_3d = False
        modelname = 'TensorModel'
        mse_predict_count = False
        X = np.zeros((len(solutions), 4))
        y = np.zeros((len(solutions), 4)) if not separate_operation_indices else np.zeros((len(solutions), 3, 4)) if label_is_3d else np.zeros((len(solutions), 3*4))
        self.y_shape = y.shape
        def f(solution):
            x = solution[0]
            y = np.zeros((3, 4)) if label_is_3d else np.zeros(3*4)
            for rule_index in range(3):
                if label_is_3d:
                    y[rule_index, solution[3][rule_index]] = 1
                else:
                    y[rule_index*4 + solution[3][rule_index]] = 1
            return x, y
        def g(solution):
            x = solution[0]
            if separate_operation_indices:
                y = np.zeros((3, 4))
                for rule_index, operation in enumerate(solution[3]):
                    if mse_predict_count:
                        y[rule_index, operation] += 1
                    else:
                        y[rule_index, operation] = 1
            else:
                y = np.zeros((4,))
                for operation in solution[3]:
                    if mse_predict_count:
                        y[operation] += 1
                    else:
                        y[operation] = 1
            return x, y
        for i, solution in enumerate(solutions):
            X[i], y[i] = g(solution) if modelname == 'MSEOperationPresence' else f(solution)
        return X, y
        # # Convert data to PyTorch tensors
        # X_tensor = torch.tensor(X)
        # y_tensor = torch.tensor(y)
        # # Create DataLoader for batching
        # dataset = TensorDataset(X_tensor, y_tensor)
        # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        # return dataloader
    def fit(self, X, y, parameters=None, num_epochs=100, batch_size=16, learning_rate=0.01, debug=False, tqdm=None):
        # Convert data to PyTorch tensors
        X_tensor = torch.tensor(X)
        y_tensor = torch.tensor(y)
        # Create DataLoader for batching
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self = self.double()
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters() if parameters is None else parameters, lr=learning_rate)
        # Training loop
        for epoch in range(num_epochs) if tqdm is None else tqdm(range(num_epochs)):
            self.train()
            total_loss = 0.0
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                # Zero the parameter gradients
                optimizer.zero_grad()
                # Forward pass
                outputs = self(inputs)
                # Compute loss
                loss = criterion(outputs, targets)
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            # Print the average loss for this epoch
            avg_loss = total_loss / len(dataloader)
            if tqdm is None and (debug or epoch == 0 or (epoch+1) % (num_epochs//5) == 0):
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        if debug:
            print("Training complete!")
        self.eval()
        return self
    def predict(self, X):
        X_tensor = torch.tensor(X)
        with torch.no_grad():
            y_pred = self(X_tensor)
        return y_pred.numpy()

class NNSingleLayer(TensorModel):
    def __init__(self, input_dim=4, hidden_dim=16, output_dim=12):
        super(NNSingleLayer, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def fit(self, X, y, num_epochs=100, batch_size=16, learning_rate=0.01, debug=False):
        return super(NNSingleLayer, self).fit(X, y, self.parameters(), num_epochs, batch_size, learning_rate, debug)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

class NNVariableWidth(TensorModel):
    def __init__(self, input_dim=4, hidden_dims=(32,32), output_dim=12):
        super(NNVariableWidth, self).__init__()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(input_dim, hidden_dims[0]))
        for i in range(1, len(hidden_dims)):
            self.fcs.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
        self.fcs.append(nn.Linear(hidden_dims[-1], output_dim))
    def fit(self, X, y, num_epochs=100, batch_size=16, learning_rate=0.01, debug=False):
        return super(NNVariableWidth, self).fit(X, y, self.parameters(), num_epochs, batch_size, learning_rate, debug)
    def forward(self, x):
        for fc in self.fcs[:-1]:
            x = F.relu(fc(x))
        x = torch.sigmoid(self.fcs[-1](x))
        return x

# modelname = 'MSEOperationPresence'
# separate_operation_indices = True # treat the 3 operations as separate probabilities
# label_is_3d = True
# # # print(solutions[0])
# # mse_predict_count = False
# # X = np.zeros((len(solutions), 4))
# # y = np.zeros((len(solutions), 4)) if not separate_operation_indices else np.zeros((len(solutions), 3, 4)) if label_is_3d else np.zeros((len(solutions), 3*4)) # predict the 3 operations using one hot encoding
# def f(solution):
#     x = solution[0]
#     y = np.zeros((3, 4)) if label_is_3d else np.zeros(3*4)
#     for rule_index in range(3):
#         if label_is_3d:
#             y[rule_index, solution[3][rule_index]] = 1
#         else:
#             y[rule_index*4 + solution[3][rule_index]] = 1
#     return x, y
# def g(solution):
#     x = solution[0]
#     if separate_operation_indices:
#         y = np.zeros((3, 4))
#         for rule_index, operation in enumerate(solution[3]):
#             if mse_predict_count:
#                 y[rule_index, operation] += 1
#             else:
#                 y[rule_index, operation] = 1
#     else:
#         y = np.zeros((4,))
#         for operation in solution[3]:
#             if mse_predict_count:
#                 y[operation] += 1
#             else:
#                 y[operation] = 1
#     return x, y
# for i, solution in enumerate(solutions):
#     X[i], y[i] = g(solution) if modelname == 'MSEOperationPresence' else f(solution) # input
# print(X.shape, X[0])
# print(y.shape, y[0])

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, TensorDataset
# import numpy as np
# # Define the MLP model
# class PytorchNN1(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(PytorchNN1, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, output_dim)
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = torch.sigmoid(self.fc2(x))  # Outputs should be between 0 and 1
#         return x
# # Set the random seed for reproducibility
# torch.manual_seed(42)
# # Define the parameters
# input_dim = 4
# hidden_dim = 16
# output_dim = 4
# learning_rate = 0.01
# num_epochs = 100
# batch_size = 16
# # Generate some dummy data for training
# num_samples = 1000
# Xfake = np.random.rand(num_samples, input_dim).astype(np.float32)  # Random input data
# yfake = np.random.randint(0, 2, (num_samples, output_dim)).astype(np.float32)  # Random binary targets
# print(X.shape, Xfake.shape)
# print(y.shape, yfake.shape)
# # Convert data to PyTorch tensors
# X_tensor = torch.tensor(X)
# y_tensor = torch.tensor(y)
# # Create DataLoader for batching
# dataset = TensorDataset(X_tensor, y_tensor)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# # Initialize the model, loss function, and optimizer
# model = PytorchNN1(input_dim, hidden_dim, output_dim).double()
# criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for multi-attribute classification
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# # Training loop
# for epoch in range(num_epochs):
#     model.train()
#     total_loss = 0.0
#     for batch_idx, (inputs, targets) in enumerate(dataloader):
#         # Zero the parameter gradients
#         optimizer.zero_grad()
#         # Forward pass
#         # inputs = inputs.float()
#         outputs = model(inputs)
#         # Compute loss
#         loss = criterion(outputs, targets)
#         # Backward pass and optimization
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     # Print the average loss for this epoch
#     avg_loss = total_loss / len(dataloader)
#     print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
# print("Training complete!")
# # Test the model on a new sample
# model.eval()
# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.neural_network import MLPClassifier, MLPRegressor
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.multioutput import MultiOutputRegressor
# from sklearn.metrics import mean_squared_error
# from sklearn.base import BaseEstimator, ClassifierMixin
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVR

# class RegressionIndexModel:
#     def __init__(self):
#         self.base_model = lambda: LogisticRegression(max_iter=1000)
    
#     def fit(self, X, y):
#         n = y.shape[1]
#         self.models = [None] * n
#         for i in range(n):
#             self.models[i] = self.base_model()
#             self.models[i] = self.models[i].fit(X, np.argmax(y[:, i, :], axis=1))
        
#         return self
    
#     def predict(self, X):
#         preds = [model.predict(X) for model in self.models] # Predict each output
#         return np.stack(preds, axis=1) # stack them together
    
#     def predict_proba(self, X):
#         probas = [model.predict_proba(X) for model in self.models]
#         return np.stack(probas, axis=1)
# class MLPCrossEntropyModel:
#     def __init__(self, propagate=True):
#         self.propagate = propagate
#         self.base_model = lambda: MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000)
#     def fit(self, X, y):
#         self.y_shape = y.shape
#         self.models = [None] * y.shape[1]
#         for i in range(y.shape[1]):
#             self.models[i] = self.base_model()
#             if self.propagate:
#                 revealed_y_data = y[:, 0:i, :]
#                 revealed_y_data = np.reshape(revealed_y_data, (revealed_y_data.shape[0], revealed_y_data.shape[1] * revealed_y_data.shape[2]))
#                 modified_x = np.concatenate((X, revealed_y_data), axis=1)
#                 self.models[i] = self.models[i].fit(modified_x, np.argmax(y[:, i, :], axis=1))
#             else:
#                 self.models[i] = self.models[i].fit(X, np.argmax(y[:, i, :], axis=1))
#         return self
    
#     def predict(self, X):
#         if self.propagate:
#             return self.predict_proba(X).argmax(axis=2)
#         else:
#             return np.swapaxes(np.array([model.predict(X) for model in self.models]), 0, 1)
    
#     def predict_proba(self, X):
#         if self.propagate:
#             predicted_proba = np.zeros((X.shape[0], self.y_shape[1], self.y_shape[2]))
#             for i in range(self.y_shape[1]):
#                 revealed_y_data = predicted_proba[:, 0:i, :]
#                 revealed_y_data = np.reshape(revealed_y_data, (revealed_y_data.shape[0], revealed_y_data.shape[1] * revealed_y_data.shape[2]))
#                 modified_x = np.concatenate((X, revealed_y_data), axis=1)
#                 out = self.models[i].predict_proba(modified_x)
#                 predicted_proba[:, i, :] = out
#             return predicted_proba
#         else:
#             return np.swapaxes(np.array([model.predict_proba(X) for model in self.models]), 0, 1)
#     def predict_operation_probabilities(self, X):
#         return self.predict_proba(X)
# class MSEOperationPresence:
#     def __init__(self, param='RandomForestRegressor'):
#         # Each class can be present or not
#         if param == 'RandomForestRegressor':
#             self.base_model = lambda: MultiOutputRegressor(RandomForestRegressor())
#         elif param == 'SVR':
#             self.base_model = lambda: MultiOutputRegressor(SVR(kernel='rbf'))
#         elif param == 'MLPRegressor':
#             self.base_model = lambda: MultiOutputRegressor(MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=1000))
#     def fit(self, X, y):
#         if separate_operation_indices:
#             self.models = [None] * y.shape[1]
#             for i in range(y.shape[1]):
#                 self.models[i] = self.base_model()
#                 self.models[i].fit(X, y[:, i, :])
#         else:
#             self.model = self.base_model()
#             self.model.fit(X, y)
#         return self
    
#     def predict(self, X):
#         if separate_operation_indices:
#             return np.stack([model.predict(X) for model in self.models], axis=1)
#         else:
#             return self.model.predict(X)

#     def predict_operation_probabilities(self, X):
#         return self.predict(X)
# class OperationAndIndex:
#     def __init__(self, param='RandomForestRegressor'):
#         # Each class can be present or not
#         if param == 'RandomForestRegressor':
#             self.base_model = lambda: MultiOutputRegressor(RandomForestRegressor())
#         elif param == 'SVR':
#             self.base_model = lambda: MultiOutputRegressor(SVR(kernel='rbf'))
#         elif param == 'MLPRegressor':
#             self.base_model = lambda: MultiOutputRegressor(MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=1000))
#     def fit(self, X, y):
#         if separate_operation_indices:
#             self.models = [None] * y.shape[1]
#             for i in range(y.shape[1]):
#                 self.models[i] = self.base_model()
#                 self.models[i].fit(X, y[:, i, :])
#         else:
#             self.model = self.base_model()
#             self.model.fit(X, y)
#         return self
    
#     def predict(self, X):
#         if separate_operation_indices:
#             return np.stack([model.predict(X) for model in self.models], axis=1)
#         else:
#             return self.model.predict(X)

#     def predict_operation_probabilities(self, X):
#         return self.predict(X)

# # First two models have issue with all models predicting the same thing
# model = {'RegressionIndexModel': RegressionIndexModel(), 'MLPCrossEntropyModel': MLPCrossEntropyModel(), 'MSEOperationPresence': MSEOperationPresence(param='RandomForestRegressor')}[modelname]
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)
# # print(y_test)
# # print(y_pred)
