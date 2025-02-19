import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from torchsummary import summary as ts

from model import ResNet18
from my_unireplknet import MyUniRepLKNet
from unireplknet_original import UniRepLKNet

def exists(item):
    return item is not None


class Trainer:
    def __init__(self, train_loader, test_loader, learning_rate, epochs, test_round, classes):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.epochs = epochs
        self.test_round = test_round

        self.train_loader = train_loader
        self.test_loader = test_loader

        # self.model = ResNet18(num_classes=classes)
        self.model = MyUniRepLKNet(classes=classes, 
                                 in_channels=3, 
                                 depth=(2,2,6,2), 
                                 dims=(40,80,160,320),
                                 drop_rate=0.,)
        # self.model = UniRepLKNet(num_classes=classes,
        #                          depths=(2, 2, 6, 2),
        #                          dims=(40,80,160,320),
        #                          kernel_sizes=((3, 3), (9, 9), (9, 9, 9, 9, 9, 9), (9, 9)))

        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=0.)
        # self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=5, T_mult=2, eta_min=1e-6)
        self.loss_func = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.test_loss_init = None


    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            total = 0
            train_loss = 0.0
            correct_predictions = 0

            # ts(model=self.model, input_size=(3,32,32), batch_size=64)

            for input, label in self.train_loader:
                input = input.to(self.device)
                label = label.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(input)
                loss = self.loss_func(output, label)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                correct_predictions += (output.argmax(dim=-1) == label).sum()
                total += input.size(0)
            
            train_acc = correct_predictions / total
            avg_loss = train_loss / len(self.train_loader)

            # print(f"Training Epoch [{epoch+1}/{self.epochs}] - Loss: {avg_loss:.4f}, Accuracy: {train_acc:.4f}, learning rate: {self.scheduler.get_last_lr()[0]:.5f}")
            # self.scheduler.step()
            print(f"Training Epoch [{epoch+1}/{self.epochs}] - Loss: {avg_loss:.4f}, Accuracy: {train_acc:.4f}")

            if (epoch % self.test_round == 0):
                test_loss = 0.0
                self.model.eval()
                test_predictions = 0
                test_total = 0

                with torch.no_grad():
                    for input, label in self.test_loader:
                        input = input.to(self.device)
                        label = label.to(self.device)

                        output = self.model(input)
                        loss = self.loss_func(output, label)
                        test_loss += loss.item()
                        test_predictions += (output.argmax(dim=-1) == label).sum()
                        test_total += input.size(0)
                    
                    test_acc = test_predictions / test_total
                    avg_test_loss = test_loss / len(self.test_loader)
                print(f"     Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
